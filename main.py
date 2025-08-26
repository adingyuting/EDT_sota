# ============
# main.py
# ============

"""Pipeline entry: RN-SMOTE → RLKF → MSGVT → IGANN (+SBOA) → Metrics + SHAP"""

# --- file: main.py ---
from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
from data_loader import load_sgcc_csv, stratified_split
from rn_smote import apply_rn_smote
from rlkf import apply_rlkf
from feature_extraction import extract_msgvt_features
from igann_model import train_igann, predict_proba, set_seed
from sboa_optimizer import sboa_optimize
from evaluate import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='outputs')
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--balance', action='store_true', help='Apply RN-SMOTE on train only')
    parser.add_argument('--drop-frac', type=float, default=0.1)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--target-ratio', type=float, default=1.0)
    parser.add_argument('--rlkf', action='store_true', help='Apply RLKF cleanup')
    parser.add_argument('--process-var', type=float, default=0.05)
    parser.add_argument('--obs-var', type=float, default=0.1)
    parser.add_argument('--use-wavelet', action='store_true', help='Use wavelet in MSGVT stats')
    parser.add_argument('--sboa', action='store_true', help='Enable SBOA hyperparam search')
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lambda-task', type=float, default=1e-4)
    parser.add_argument('--lambda-bg', type=float, default=1e-5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Load
    X, y = load_sgcc_csv(args.data_dir)

    # Split first to avoid leakage
    X_tr_raw, X_te_raw, y_tr, y_te = stratified_split(X, y, test_size=args.test_size, random_state=args.seed)

    # Balance (train only)
    if args.balance:
        Xb, yb = apply_rn_smote(X_tr_raw, y_tr, minority_label=1, k=args.k, drop_frac=args.drop_frac, target_ratio=args.target_ratio, random_state=args.seed)
    else:
        Xb, yb = X_tr_raw, y_tr

    # RLKF cleanup
    if args.rlkf:
        Xb = apply_rlkf(Xb, process_var=args.process_var, obs_var=args.obs_var)
        X_te_proc = apply_rlkf(X_te_raw, process_var=args.process_var, obs_var=args.obs_var)
    else:
        X_te_proc = X_te_raw

    # MSGVT-like stats
    X_tr_feat = extract_msgvt_features(Xb, use_wavelet=args.use_wavelet)
    X_te_feat = extract_msgvt_features(X_te_proc, use_wavelet=args.use_wavelet)

    # Normalize features (z-score fit on train)
    mu = X_tr_feat.mean(axis=0)
    sigma = X_tr_feat.std(axis=0) + 1e-12
    X_trn = (X_tr_feat - mu) / sigma
    X_ten = (X_te_feat - mu) / sigma

    # Train IGANN (+ optional SBOA)
    lam_task, lam_bg = args.lambda_task, args.lambda_bg
    if args.sboa:
        lam_task, lam_bg = sboa_optimize(X_trn, yb, hidden=args.hidden, lr=args.lr, max_epochs=100, iterations=20, pop_size=16, seed=args.seed)

    model = train_igann(X_trn, yb, X_ten, y_te, lr=args.lr, max_epochs=args.epochs, hidden=args.hidden,
                        lambda_task=lam_task, lambda_bg=lam_bg, patience=args.patience, seed=args.seed)

    # Predict & Metrics
    proba = predict_proba(model, X_ten)
    metrics = compute_metrics(y_te, proba)
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump({k: (float(v) if not isinstance(v, (int, str)) else v) for k, v in metrics.items()}, f, indent=2)

    # Save feature importance via linear_a (as a quick interpretability proxy)
    import torch
    lin_a = model.linear_a.detach().cpu().numpy()
    feat_names = ['Entropy','Mean','Variance','Kurtosis','Correlation','Contrast']
    pd.DataFrame({'feature': feat_names, 'linear_weight': lin_a}).to_csv(os.path.join(args.out_dir, 'igann_linear_weights.csv'), index=False)

    # Print summary
    print("=== IGANN Evaluation (Test) ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v}")
    print(f"Chosen λ_task={lam_task:.2e}, λ_bg={lam_bg:.2e}")


if __name__ == '__main__':
    main()