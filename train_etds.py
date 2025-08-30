
import os, json, argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from data import (
    load_sgcc_csv,
    train_test_split_stratified,
)
from sldi import SLDIImputer
from vmd_features import batch_vmd
from ram_bigru import build_ram_bigru_model
from metrics import binary_metrics

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def train_binary_sgcc(
    feature_csv: str,
    label_csv: str,
    outdir: str,
    train_sizes=(0.5, 0.6, 0.7, 0.8),
    vmd_K: int = 6,
    vmd_alpha: float = 2000,
    vmd_tau: float = 0.0,
    vmd_tol: float = 1e-7,
    attn_units: int = 32,
    bigru_units: int = 200,
    xi: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
):
    os.makedirs(outdir, exist_ok=True)
    set_seed(seed)

    print("[1/4] Loading CSV features and labels ...")
    X, y = load_sgcc_csv(feature_csv, label_csv)
    print(f"    Loaded X{X.shape}, y{y.shape}")

    print("[2/4] Imputing missing values with SLDI ...")
    print(f"    Original shape: {X.shape}")
    sldi = SLDIImputer(max_iter=500, random_state=seed)
    X_imp = sldi.fit_transform(X)
    print(f"    After imputation: {X_imp.shape}")
    print(f"    Imputed data statistics: Mean = {np.mean(X_imp):.4f}, Std = {np.std(X_imp):.4f}")

    print("[3/4] Extracting VMD features ...")
    print(f"    Performing VMD on signal with K={vmd_K}, alpha={vmd_alpha}, tol={vmd_tol}.")
    modes = batch_vmd(X_imp, K=vmd_K, alpha=vmd_alpha, tau=vmd_tau, tol=vmd_tol)
    print(f"    VMD decomposition results: {modes.shape}")
    X_tensor = np.transpose(modes, (0, 2, 1)).astype(np.float32)

    results = {}
    for train_size in train_sizes:
        print(f"[4/4] Training with {int(train_size*100)}% training data and {int((1-train_size)*100)}% testing data.")
        Xtr_full, Xte, ytr_full, yte = train_test_split_stratified(
            X_tensor, y, test_size=1-train_size, random_state=seed
        )
        print(f"    Train set: {Xtr_full.shape}, Test set: {Xte.shape}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_metrics = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(Xtr_full, ytr_full), 1):
            print(f"    Training fold {fold} with {int(train_size*100)}% training data.")
            Xtr, Xval = Xtr_full[tr_idx], Xtr_full[val_idx]
            ytr, yval = ytr_full[tr_idx], ytr_full[val_idx]
            T, M = Xtr.shape[1], Xtr.shape[2]
            print("    Using BiGRU with 200 units and ReLU activation.")
            model = build_ram_bigru_model(T=T, M=M, n_classes=2, attn_units=attn_units, xi=xi, bigru_units=bigru_units)
            print(f"    Using Adamax optimizer with lr={lr}, beta_1=0.9, beta_2=0.999.")
            opt = tf.keras.optimizers.Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999)
            model.compile(optimizer=opt, loss="binary_crossentropy")
            model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size, verbose=0)
            y_prob = model.predict(Xval, verbose=0)[:, 0]
            met = binary_metrics(yval, y_prob)
            print(
                f"    Fold {fold} metrics: F1={met['F1']:.4f}, AUC={met['AUC']:.4f}, "
                f"Recall={met['RE']:.4f}, FPR={met['FPR']:.4f}"
            )
            fold_metrics.append(met)

        avg_met = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        print(
            f"    Average 5-fold metrics: F1={avg_met['F1']:.4f}, AUC={avg_met['AUC']:.4f}, "
            f"Recall={avg_met['RE']:.4f}, FPR={avg_met['FPR']:.4f}"
        )

        # Train on full training set and evaluate on held-out test set
        T, M = Xtr_full.shape[1], Xtr_full.shape[2]
        model = build_ram_bigru_model(T=T, M=M, n_classes=2, attn_units=attn_units, xi=xi, bigru_units=bigru_units)
        opt = tf.keras.optimizers.Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=opt, loss="binary_crossentropy")
        model.fit(Xtr_full, ytr_full, epochs=epochs, batch_size=batch_size, verbose=0)
        y_prob = model.predict(Xte, verbose=0)[:, 0]
        test_met = binary_metrics(yte, y_prob)
        print(
            f"    Test metrics: F1={test_met['F1']:.4f}, AUC={test_met['AUC']:.4f}, "
            f"Recall={test_met['RE']:.4f}, FPR={test_met['FPR']:.4f}"
        )
        results[f"train_{int(train_size*100)}"] = test_met
        with open(os.path.join(outdir, f"metrics_{int(train_size*100)}.json"), "w") as f:
            json.dump(test_met, f, indent=2)

    return results

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
    parser.add_argument("--vmd-K", type=int, default=6)
    parser.add_argument("--vmd-alpha", type=float, default=2000)
    parser.add_argument("--vmd-tau", type=float, default=0.0)
    parser.add_argument("--vmd-tol", type=float, default=1e-7)
    parser.add_argument("--attn-units", type=int, default=32)
    parser.add_argument("--bigru-units", type=int, default=200)
    parser.add_argument("--xi", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    met = train_binary_sgcc(
        feature_csv=args.features,
        label_csv=args.labels,
        outdir=args.outdir,
        vmd_K=args.vmd_K,
        vmd_alpha=args.vmd_alpha,
        vmd_tau=args.vmd_tau,
        vmd_tol=args.vmd_tol,
        attn_units=args.attn_units,
        bigru_units=args.bigru_units,
        xi=args.xi,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    print(json.dumps(met, indent=2))

if __name__ == "__main__":
    main()
