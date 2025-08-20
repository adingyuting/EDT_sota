import argparse, os, random, json
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, recall_score

############################################################
# Utils
############################################################

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def minmax_norm(x: np.ndarray, eps=1e-8) -> np.ndarray:
    # x: [U,D,48] already user-normalized in preprocessing; keep as-is or re-normalize per user
    return x

def windowize(user_days_48: np.ndarray, window_len: int = 336, stride: int = 48) -> np.ndarray:
    T = user_days_48.shape[0] * 48
    series = user_days_48.reshape(-1)
    if T < window_len: return np.empty((0, window_len), dtype=np.float32)
    out = []
    for s in range(0, T - window_len + 1, stride):
        out.append(series[s:s+window_len].astype(np.float32))
    return np.stack(out, axis=0)

def make_user_splits(num_users: int, ratio=(0.8, 0.1, 0.1)):
    idx = np.arange(num_users); np.random.shuffle(idx)
    a = int(ratio[0]*num_users); b = int(ratio[1]*num_users)
    return idx[:a], idx[a:a+b], idx[a+b:]

############################################################
# Self-supervised masking (for pretraining)
############################################################

def build_mask(L: int, r: float = 0.3, l0: float = 8.0) -> np.ndarray:
    l1 = ((1 - r) / r) * l0
    p0 = 1.0 / max(1.0, l0); p1 = 1.0 / max(1.0, l1)
    vals = []; total = 0; cur_zero = True
    while total < L:
        import numpy as _np
        p = p0 if cur_zero else p1
        seg = int(_np.random.geometric(p)); seg = max(1, seg)
        if total + seg > L: seg = L - total
        vals.extend([0 if cur_zero else 1]*seg)
        total += seg; cur_zero = not cur_zero
    return np.array(vals, dtype=np.float32)

############################################################
# Dataset (real labels; no synthetic attacks)
############################################################

class SGCCWindows(Dataset):
    def __init__(self, daily48: np.ndarray, user_labels: np.ndarray, users_idx: np.ndarray,
                 split: str, window_len: int = 336, for_pretrain: bool = False,
                 labeled_frac: float = 0.10, seed: int = 42):
        assert split in ["pretrain","train","val","test"]
        set_seed(seed)
        self.samples, self.labels = [], []
        self.L = window_len
        for uid in users_idx:
            u = daily48[uid]
            W = windowize(u, window_len=window_len, stride=48)
            if W.size == 0: continue
            y_user = int(user_labels[uid])  # 用户级标签
            if split == "pretrain" and for_pretrain:
                # 仅做无监督：全部窗口记为 unlabeled(-1)
                for i in range(W.shape[0]):
                    self.samples.append(W[i]); self.labels.append(-1)
            else:
                # 有监督：用真实标签
                # 若需要子采样 labeled_frac，可在 train split 上抽样；val/test 全保留
                if split == "train" and 0.0 < labeled_frac < 1.0:
                    import numpy as _np
                    n = W.shape[0]
                    k = max(1, int(labeled_frac * n))
                    choose = _np.random.choice(n, size=k, replace=False)
                    for i in choose:
                        self.samples.append(W[i]); self.labels.append(y_user)
                else:
                    for i in range(W.shape[0]):
                        self.samples.append(W[i]); self.labels.append(y_user)
        self.samples = np.stack(self.samples).astype(np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

############################################################
# Model: HPTE + heads
############################################################

class HierarchicalPatchEmbedding(nn.Module):
    def __init__(self, d: int, patch_size: int):
        super().__init__(); self.d=d; self.p=patch_size; self.proj=nn.Linear(self.p*d, d)
    def forward(self, x):
        B,L,d = x.shape; assert L%self.p==0
        x = x.view(B, L//self.p, self.p*d); return self.proj(x)

class HPTELayer(nn.Module):
    def __init__(self, d: int, heads: int, patch_size: int, ff_mult: int=4, dropout: float=0.0):
        super().__init__()
        self.hpe = HierarchicalPatchEmbedding(d, patch_size)
        self.mha = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff_mult*d,
                                              batch_first=True, dropout=dropout, activation="gelu")
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult*d), nn.GELU(), nn.Linear(ff_mult*d, d))
    def forward(self, x):
        h = self.hpe(x); y = self.mha(h); y = self.ln1(h + y); z = self.ff(y); return self.ln2(y + z)

class MFFTD(nn.Module):
    def __init__(self, input_dim=1, d=128, heads=8, patch_sizes=[2,3,8], ff_mult=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d)
        self.layers = nn.ModuleList([HPTELayer(d, heads, p, ff_mult) for p in patch_sizes])
        self.d=d; self.patch_sizes=patch_sizes
        self.pretrain_head=None; self.classifier=None
    def forward_features(self, x):
        if x.dim()==2: x=x.unsqueeze(-1)
        x = self.input_proj(x)
        for layer in self.layers: x = layer(x)
        return x
    def _Lf(self, L):
        for p in self.patch_sizes: L//=p
        return L
    def make_pretrain_head(self, L_in): self.pretrain_head = nn.Linear(self._Lf(L_in)*self.d, L_in)
    def make_classifier(self, L_in): self.classifier = nn.Linear(self._Lf(L_in)*self.d, 1)
    def forward_pretrain(self, x):
        B,L = x.shape; feats=self.forward_features(x); return self.pretrain_head(feats.reshape(B,-1))
    def forward_classify(self, x):
        B,L = x.shape; feats=self.forward_features(x); return self.classifier(feats.reshape(B,-1)).squeeze(-1)

############################################################
# Train / Eval
############################################################

def evaluate(model, loader, device):
    model.eval(); ys, ps = [], []
    import numpy as _np
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); logit = model.forward_classify(x)
            prob = torch.sigmoid(logit)
            ys.append(y.numpy()); ps.append(prob.detach().cpu().numpy())
    ys = _np.concatenate(ys).astype(int); ps = _np.concatenate(ps).astype(float)
    preds = (ps >= 0.5).astype(int)
    f1 = f1_score(ys, preds, zero_division=0)
    try: auc = roc_auc_score(ys, ps)
    except Exception: auc = float('nan')
    rec = recall_score(ys, preds, zero_division=0)
    tn = ((ys==0)&(preds==0)).sum(); fp = ((ys==0)&(preds==1)).sum()
    fpr = fp / (fp + tn + 1e-8)
    return f1, float(auc), rec, fpr

# Pretraining: use unlabeled windows only (no synthetic attacks)

def train_pretrain(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(args.data, allow_pickle=True)
    daily = data['daily_kwh'].astype(np.float32)
    user_labels = data['user_labels'].astype(int)
    U = daily.shape[0]
    tr_idx, va_idx, _ = make_user_splits(U, ratio=(0.8,0.2,0.0))

    tr_ds = SGCCWindows(daily, user_labels, tr_idx, split='pretrain', window_len=args.window_len, for_pretrain=True, seed=args.seed)
    va_ds = SGCCWindows(daily, user_labels, va_idx, split='train', window_len=args.window_len, for_pretrain=False, seed=args.seed)
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = MFFTD(d=128, heads=8, patch_sizes=[2,3,8]).to(device)
    model.make_pretrain_head(args.window_len)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def make_mask(B,L):
        import numpy as _np
        return torch.tensor(_np.stack([build_mask(L, args.mask_ratio, args.mask_l0) for _ in range(B)]), dtype=torch.float32, device=device)

    best = 1e9
    for ep in range(args.epochs):
        model.train(); s=0.0; n=0
        for x, _ in tr:
            x = x.to(device); M = make_mask(x.size(0), x.size(1)); xm = x*M
            pred = model.forward_pretrain(xm); mask = (M==0).float()
            mse = ((pred-x)**2*mask).sum()/(mask.sum()+1e-8)
            opt.zero_grad(); mse.backward(); opt.step()
            s+=mse.item(); n+=1
        # quick val as reconstruction
        model.eval(); vs=0.0; c=0
        with torch.no_grad():
            for x,_ in va:
                x = x.to(device); M=make_mask(x.size(0), x.size(1)); xm=x*M
                pred=model.forward_pretrain(xm); mask=(M==0).float()
                mse=((pred-x)**2*mask).sum()/(mask.sum()+1e-8)
                vs+=mse.item(); c+=1
        vs/=max(1,c)
        print(f"[Pretrain] {ep+1}/{args.epochs} train_mse={s/max(1,n):.4f} val_mse={vs:.4f}")
        if vs<best:
            best=vs; os.makedirs(os.path.dirname(args.out), exist_ok=True); torch.save(model.state_dict(), args.out)

# Finetune: supervised on real labels (optionally semi-supervised via labeled_frac)

def train_finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(args.data, allow_pickle=True)
    daily = data['daily_kwh'].astype(np.float32)
    user_labels = data['user_labels'].astype(int)
    U = daily.shape[0]
    tr_idx, va_idx, _ = make_user_splits(U, ratio=(0.8,0.2,0.0))

    tr_ds = SGCCWindows(daily, user_labels, tr_idx, split='train', window_len=args.window_len, for_pretrain=False, labeled_frac=args.labeled_frac, seed=args.seed)
    va_ds = SGCCWindows(daily, user_labels, va_idx, split='val', window_len=args.window_len, for_pretrain=False, seed=args.seed)
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = MFFTD(d=128, heads=8, patch_sizes=[2,3,8]).to(device)
    model.make_pretrain_head(args.window_len)
    model.make_classifier(args.window_len)
    if args.pretrained and os.path.exists(args.pretrained):
        sd=torch.load(args.pretrained, map_location=device); model.load_state_dict(sd, strict=False)

    # compute pos_weight from labeled train set
    y_tr = tr_ds.labels[tr_ds.labels>=0]
    pos = (y_tr==1).sum(); neg = (y_tr==0).sum()
    pos_weight = float(neg/max(1,pos)) if args.pos_weight<=0 else args.pos_weight
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # Stage-1: freeze FE
    fe_params = list(model.input_proj.parameters()) + [p for l in model.layers for p in l.parameters()]
    for p in fe_params: p.requires_grad=False
    opt1 = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=1e-3)

    for ep in range(args.epochs1):
        model.train()
        for x,y in tr:
            x=x.to(device); yb=y.to(device).float()
            logit=model.forward_classify(x); loss=bce(logit,yb)
            opt1.zero_grad(); loss.backward(); opt1.step()
        f1,auc,rec,fpr = evaluate(model, va, device)
        print(f"[Stage1] {ep+1}/{args.epochs1} F1={f1:.3f} AUC={auc:.3f} Recall={rec:.3f} FPR={fpr:.3f}")

    # Stage-2: unfreeze
    for p in fe_params: p.requires_grad=True
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-3)
    best=0.0
    for ep in range(args.epochs2):
        model.train()
        for x,y in tr:
            x=x.to(device); yb=y.to(device).float()
            logit=model.forward_classify(x); loss=bce(logit,yb)
            opt2.zero_grad(); loss.backward(); opt2.step()
        f1,auc,rec,fpr = evaluate(model, va, device)
        print(f"[Stage2] {ep+1}/{args.epochs2} F1={f1:.3f} AUC={auc:.3f} Recall={rec:.3f} FPR={fpr:.3f}")
        if f1>best: best=f1; os.makedirs(os.path.dirname(args.out), exist_ok=True); torch.save(model.state_dict(), args.out)

# Eval on held-out test users

def evaluate_ckpt(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(args.data, allow_pickle=True)
    daily = data['daily_kwh'].astype(np.float32); user_labels = data['user_labels'].astype(int)
    U = daily.shape[0]
    _, _, te_idx = make_user_splits(U, ratio=(0.8,0.1,0.1))
    te_ds = SGCCWindows(daily, user_labels, te_idx, split='test', window_len=args.window_len, seed=args.seed)
    te = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = MFFTD(d=128, heads=8, patch_sizes=[2,3,8]).to(device)
    model.make_pretrain_head(args.window_len); model.make_classifier(args.window_len)
    sd=torch.load(args.ckpt, map_location=device); model.load_state_dict(sd, strict=False)
    f1,auc,rec,fpr = evaluate(model, te, device)
    print(json.dumps({"F1":f1, "AUC":auc, "Recall":rec, "FPR":fpr}, indent=2))

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='cmd', required=True)

    a=sub.add_parser('pretrain')
    a.add_argument('--data', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--window_len', type=int, default=336)
    a.add_argument('--batch_size', type=int, default=64)
    a.add_argument('--epochs', type=int, default=50)
    a.add_argument('--mask_ratio', type=float, default=0.3)
    a.add_argument('--mask_l0', type=float, default=8.0)
    a.add_argument('--seed', type=int, default=42)

    b=sub.add_parser('finetune')
    b.add_argument('--data', required=True)
    b.add_argument('--pretrained', default='')
    b.add_argument('--out', required=True)
    b.add_argument('--window_len', type=int, default=336)
    b.add_argument('--batch_size', type=int, default=64)
    b.add_argument('--epochs1', type=int, default=20)
    b.add_argument('--epochs2', type=int, default=80)
    b.add_argument('--labeled_frac', type=float, default=0.10, help='fraction of train windows to label (semi-supervised)')
    b.add_argument('--pos_weight', type=float, default=-1.0, help='<=0 to auto-compute from data')
    b.add_argument('--seed', type=int, default=42)

    c=sub.add_parser('eval')
    c.add_argument('--data', required=True)
    c.add_argument('--ckpt', required=True)
    c.add_argument('--window_len', type=int, default=336)
    c.add_argument('--batch_size', type=int, default=128)
    c.add_argument('--seed', type=int, default=42)

    args=p.parse_args(); set_seed(args.seed)
    if args.cmd=='pretrain': train_pretrain(args)
    elif args.cmd=='finetune': train_finetune(args)
    else: evaluate_ckpt(args)