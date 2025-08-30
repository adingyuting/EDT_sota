
import os, time, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data import (
    load_sgcc_csv,
    train_test_split_stratified,
    standardize_per_series,
    make_tensor_data_from_modes,
)
from sldi import SLDIImputer
from vmd_features import batch_vmd
from ram_bigru import build_ram_bigru_model
from metrics import binary_metrics, BinaryMetricsCallback

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def train_binary_sgcc(
    feature_csv: str,
    label_csv: str,
    outdir: str,
    test_size: float = 0.3,
    vmd_K: int = 6,
    attn_units: int = 32,
    bigru_units: int = 64,
    reg_weight: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
):
    os.makedirs(outdir, exist_ok=True)
    set_seed(seed)

    print("[1/9] Loading CSV features and labels ...")
    X, y = load_sgcc_csv(feature_csv, label_csv)
    print(f"    Loaded X{X.shape}, y{y.shape}")

    print("[2/9] Imputing missing values with SLDI ...")
    sldi = SLDIImputer(max_iter=500, random_state=seed)
    X_imp = sldi.fit_transform(X)

    print("[3/9] Standardizing per series ...")
    # X_imp = standardize_per_series(X_imp)

    print("[4/9] Extracting VMD features ...")
    modes = batch_vmd(X_imp, K=vmd_K)    # (N, K, T)
    X_tensor = make_tensor_data_from_modes(modes, stack=True)  # (N, T, 4)

    print("[5/9] Splitting train/test/validation sets ...")
    Xtr_full, Xte, ytr_full, yte = train_test_split_stratified(
        X_tensor, y, test_size=test_size, random_state=seed
    )
    Xtr, Xval, ytr, yval = train_test_split_stratified(
        Xtr_full, ytr_full, test_size=0.2, random_state=seed
    )
    T = Xtr.shape[1]
    M = Xtr.shape[2]
    n_classes = 2

    print("[6/9] Building RAM-BiGRU model ...")
    model = build_ram_bigru_model(T=T, M=M, n_classes=n_classes, attn_units=attn_units, reg_weight=reg_weight, bigru_units=bigru_units)
    opt = tf.keras.optimizers.Adamax(learning_rate=lr)
    loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    print("[7/9] Training ...")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        BinaryMetricsCallback(val_data=(Xval, yval)),
    ]
    t0 = time.time()
    hist = model.fit(
        Xtr,
        ytr,
        validation_data=(Xval, yval),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    train_time = time.time() - t0
    print(f"    Training finished in {train_time:.2f} sec")

    print("[8/9] Evaluating on test set ...")
    prob = model.predict(Xte, batch_size=batch_size, verbose=0)
    y_prob = prob[:, 0]
    met = binary_metrics(yte, y_prob)
    met["ElapsedTime_sec"] = float(train_time)

    print("[9/9] Saving artifacts to", outdir)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(met, f, indent=2)
    model.save(os.path.join(outdir, "ram_bigru_tf"))
    with open(os.path.join(outdir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in hist.history.get(k, [])] for k in hist.history}, f, indent=2)

    return met

def main():
    parser = argparse.ArgumentParser(
        description="ETDS (SLDI + VMD + RAM-BiGRU + Adamax) Reproduction"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="./data/features.csv",
        help="Path to feature.csv  (N x T)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="./data/labels.csv",
        help="Path to label.csv    (N,) 0/1",
    )
    parser.add_argument("--outdir", type=str, default="./outputs_sgcc")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--vmd-K", type=int, default=6)
    parser.add_argument("--attn-units", type=int, default=32)
    parser.add_argument("--bigru-units", type=int, default=64)
    parser.add_argument("--reg-weight", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    met = train_binary_sgcc(
        feature_csv=args.features,
        label_csv=args.labels,
        outdir=args.outdir,
        test_size=args.test_size,
        vmd_K=args.vmd_K,
        attn_units=args.attn_units,
        bigru_units=args.bigru_units,
        reg_weight=args.reg_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    print(json.dumps(met, indent=2))

if __name__ == "__main__":
    main()
