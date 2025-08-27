# ===========
# main.py
# ===========

"""Train models across multiple train/test splits and report metrics.

Supports the original IGANN pipeline as well as a simple 1D CNN baseline
that operates directly on the cleaned time-series. The CNN option allows
experiments with a higher-capacity model that typically reaches higher
AUC (≈0.78) on this task.
"""

from __future__ import annotations
import os, json, argparse
import numpy as np
from data_loader import load_sgcc_csv, stratified_split
from rn_smote import apply_rn_smote
from rlkf import apply_rlkf
from feature_extraction import extract_msgvt_features
from igann_model import train_igann, set_seed
from cnn_model import train_cnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=r"D:\\学术工作\\pythonProject\\Igann_Etd\\data")
    parser.add_argument('--out-dir', type=str, default='outputs')
    parser.add_argument('--train-ratios', type=str, default='0.5,0.6,0.7,0.8',
                        help='Comma separated list of training set ratios')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--balance', action='store_true', help='Apply RN-SMOTE on train only')
    parser.add_argument('--drop-frac', type=float, default=0.1)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--target-ratio', type=float, default=0.7,
                        help='Desired minority/majority ratio after RN-SMOTE')
    parser.add_argument('--process-var', type=float, default=0.05)
    parser.add_argument('--obs-var', type=float, default=0.1)
    parser.add_argument('--use-wavelet', action='store_true', help='Use wavelet in MSGVT stats')
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lambda-task', type=float, default=1e-4)
    parser.add_argument('--lambda-bg', type=float, default=1e-5)
    parser.add_argument('--model', type=str, default='igann', choices=['igann', 'cnn'],
                        help='Choose model: igann or cnn baseline')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    print("Loading data...")
    X, y = load_sgcc_csv(args.data_dir)

    train_ratios = [float(r) for r in args.train_ratios.split(',')]
    results = {}

    for tr_ratio in train_ratios:
        print(f"\n==== Train ratio {tr_ratio:.1f} ====")
        test_size = 1.0 - tr_ratio
        X_train_full, X_te_raw, y_train_full, y_te = stratified_split(
            X, y, test_size=test_size, random_state=args.seed
        )

        # hold out validation from raw training portion
        X_tr_raw, X_val_raw, y_tr, y_val = stratified_split(
            X_train_full, y_train_full, test_size=0.2, random_state=args.seed
        )

        print("Applying RLKF cleanup...")
        X_tr_proc = apply_rlkf(X_tr_raw, process_var=args.process_var, obs_var=args.obs_var)
        X_val_proc = apply_rlkf(X_val_raw, process_var=args.process_var, obs_var=args.obs_var)
        X_te_proc  = apply_rlkf(X_te_raw,  process_var=args.process_var, obs_var=args.obs_var)

        pos_rate = float((y_tr == 1).mean())
        if args.balance or pos_rate < 0.2:
            if not args.balance:
                print(f"Positive rate {pos_rate:.3f} <0.2; applying RN-SMOTE automatically")
            else:
                print("Applying RN-SMOTE...")
            X_tr_bal, y_tr_bal = apply_rn_smote(
                X_tr_proc, y_tr, minority_label=1, k=args.k,
                drop_frac=args.drop_frac, target_ratio=args.target_ratio,
                random_state=args.seed
            )
        else:
            print("Skipping RN-SMOTE...")
            X_tr_bal, y_tr_bal = X_tr_proc, y_tr

        if args.model == 'igann':
            print("Extracting MSGVT features...")
            X_tr_feat = extract_msgvt_features(X_tr_bal, use_wavelet=args.use_wavelet)
            X_val_feat = extract_msgvt_features(X_val_proc, use_wavelet=args.use_wavelet)
            X_te_feat  = extract_msgvt_features(X_te_proc,  use_wavelet=args.use_wavelet)

            print("Normalizing features...")
            mu = X_tr_feat.mean(axis=0)
            sigma = X_tr_feat.std(axis=0) + 1e-12
            X_trn = (X_tr_feat - mu) / sigma
            X_valn = (X_val_feat - mu) / sigma
            X_ten  = (X_te_feat  - mu) / sigma

            print("Training IGANN model...")
            model, metrics = train_igann(
                X_trn, y_tr_bal, X_valn, y_val, X_ten, y_te,
                lr=args.lr, max_epochs=args.epochs, hidden=args.hidden,
                lambda_task=args.lambda_task, lambda_bg=args.lambda_bg,
                patience=args.patience, seed=args.seed
            )
        else:
            print("Normalizing sequences...")
            mu = X_tr_bal.mean(axis=0, keepdims=True)
            sigma = X_tr_bal.std(axis=0, keepdims=True) + 1e-12
            X_trn = (X_tr_bal - mu) / sigma
            X_valn = (X_val_proc - mu) / sigma
            X_ten  = (X_te_proc  - mu) / sigma

            print("Training CNN model...")
            model, metrics = train_cnn(
                X_trn, y_tr_bal, X_valn, y_val, X_ten, y_te,
                lr=args.lr, max_epochs=args.epochs,
                patience=args.patience, seed=args.seed
            )

        print("=== Best Metrics (Test) ===")
        for k, v in metrics.items():
            print(f"{k:>9}: {v}")
        results[f"train_ratio_{tr_ratio:.1f}"] = metrics

    with open(os.path.join(args.out_dir, 'metrics_by_ratio.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()

