#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SGCC 预处理（真实标签版）：
- 输入可以是：
  (A) 包含 [user_id, timestamp, kwh] 的“长表” CSV（或包含多个此类 CSV 的文件夹），或
  (B) 仅含 [user_id, t1, t2, ...] 的“宽表”矩阵 CSV，其中后续列为按时间顺序排列的用电量。
  标签可来自单独 CSV (--labels_csv)，格式为 [user_id, label]。
- 输出：data/sgcc_daily_48.npz，包含：
  - daily_kwh: [U, D, 48] 逐用户日48点矩阵（按用户 min–max 归一化；若时间点非48倍数，尾部填0）；
  - users: [U] user_id 列表；
  - user_labels: [U] 用户级标签（0/1）。

用法示例：
python data_pre.py \
  --in /path/to/feature_or_long_csv \
  --out data/sgcc_daily_48.npz \
  --start 2014-01-01 --end 2016-10-31 \
  --freq 30min --fill none [--labels_csv /path/to/user_labels.csv]
"""
import argparse
import pandas as pd
import numpy as np
import glob, os

LABEL_CANDIDATES = ["label", "is_theft", "fraud", "class", "y"]

def read_any(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSVs under {path}")
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    uid_col = cols.get("user_id") or cols.get("uid") or cols.get("customer_id")
    ts_col  = cols.get("timestamp") or cols.get("time") or cols.get("datetime")
    val_col = cols.get("kwh") or cols.get("value") or cols.get("consumption") or cols.get("energy")
    if not (uid_col and ts_col and val_col):
        raise ValueError(f"Expected columns like user_id/timestamp/kwh; got {list(df.columns)}")
    df = df[[uid_col, ts_col, val_col]].rename(columns={uid_col:"user_id", ts_col:"timestamp", val_col:"kwh"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["timestamp", "kwh", "user_id"]).reset_index(drop=True)
    return df

def read_feature_matrix(path):
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("feature matrix must contain user_id and time columns")
    users = df.iloc[:, 0].tolist()
    values = df.iloc[:, 1:].astype(float).values
    # per-user min–max normalization
    mn = values.min(axis=1, keepdims=True)
    mx = values.max(axis=1, keepdims=True)
    diff = mx - mn
    diff[diff < 1e-8] = 1.0
    values = (values - mn) / diff
    # pad to multiples of 48
    T = values.shape[1]
    pad = (-T) % 48
    if pad:
        values = np.pad(values, ((0, 0), (0, pad)), mode="constant")
        T += pad
    D = T // 48
    data = values.reshape(len(users), D, 48)
    return users, data

def read_labels(labels_csv, raw_df=None):
    if labels_csv:
        ldf = pd.read_csv(labels_csv)
        cols = {c.lower(): c for c in ldf.columns}
        uid = cols.get("user_id") or cols.get("uid") or cols.get("customer_id")
        lab = None
        for k in LABEL_CANDIDATES:
            lab = lab or cols.get(k)
        if not (uid and lab):
            raise ValueError("labels_csv must have user_id and label column")
        out = ldf[[uid, lab]].rename(columns={uid:"user_id", lab:"label"})
        out["label"] = (out["label"].astype(float) > 0.5).astype(int)
        return out
    else:
        # 尝试从原始明细中抓取标签（若存在）→ 聚合到用户级（任意一条为1则该用户为1）
        if raw_df is None:
            return None
        found = None
        for k in LABEL_CANDIDATES:
            if k in map(str.lower, raw_df.columns):
                # 重新读一次以免我们之前丢了该列
                found = k
                break
        if found is None:
            return None
        # 若原始文件含标签，建议用户直接提供 labels_csv 更稳妥。
        raise NotImplementedError("Detected potential label columns in raw CSVs; please supply --labels_csv explicitly for clarity.")

def build_daily_48(df, start, end, freq="30min", fill="none", drop_threshold=0.2):
    full_idx = pd.date_range(start=start, end=end, freq=freq, inclusive="both")
    arrs, users, present_days = [], [], None
    for uid, g in df.groupby("user_id"):
        g = g.sort_values("timestamp")
        s = g.set_index("timestamp")["kwh"].astype(float)
        s = s.resample(freq).sum()
        s = s.reindex(full_idx)
        if fill == "zero":
            s = s.fillna(0.0)
        elif fill == "ffill":
            s = s.ffill().bfill()
        elif fill == "none":
            pass
        else:
            raise ValueError("fill must be one of: none, zero, ffill")
        df_user = s.to_frame("kwh")
        df_user["date"] = df_user.index.date
        df_user["slot"] = df_user.index.time
        pivot = df_user.pivot_table(index="date", columns="slot", values="kwh", aggfunc="first")
        if pivot.shape[1] != 48:
            continue
        mask_complete = pivot.notna().sum(axis=1) == 48
        pivot = pivot[mask_complete]
        # 过滤无效用户：至少保留 (1 - drop_threshold) * 全期天数
        total_days = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
        if len(pivot) < (1 - drop_threshold) * total_days:
            continue
        vals = pivot.fillna(0.0).values.astype(np.float32)
        mn, mx = vals.min(), vals.max()
        if mx - mn > 1e-8:
            vals = (vals - mn) / (mx - mn)
        arrs.append(vals)
        users.append(uid)
        # 记录可用天的交集（可选：此处用最小长度对齐）
    if not arrs:
        raise RuntimeError("No valid users produced complete daily-48 matrices")
    min_days = min(a.shape[0] for a in arrs)
    arrs = [a[:min_days] for a in arrs]
    data = np.stack(arrs, axis=0)  # [U, D, 48]
    return users, data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels_csv", default=None, help="CSV with [user_id,label]")
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2016-10-31")
    ap.add_argument("--freq", default="30min")
    ap.add_argument("--fill", default="none", choices=["none","zero","ffill"])
    ap.add_argument("--drop_threshold", type=float, default=0.2)
    args = ap.parse_args()

    preview = pd.read_csv(args.input_path, nrows=1)
    cols = [c.lower() for c in preview.columns]
    if any(c in cols for c in ["timestamp", "time", "datetime"]):
        df = read_any(args.input_path)
        users, data = build_daily_48(df, args.start, args.end, args.freq, args.fill, args.drop_threshold)
    else:
        users, data = read_feature_matrix(args.input_path)

    # 读取用户级标签
    lab_df = read_labels(args.labels_csv)
    if lab_df is None:
        raise SystemExit("Please provide --labels_csv (user_id,label). Real-label SGCC requires explicit labels.")
    # 对齐 users 顺序
    lab_map = {r.user_id: int(r.label) for _, r in lab_df.iterrows()}
    user_labels = np.array([lab_map.get(u, 0) for u in users], dtype=np.int64)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, daily_kwh=data, users=np.array(users, dtype=object), user_labels=user_labels)
    print(f"Saved daily_kwh{data.shape}, labels{user_labels.shape} to {args.out}")
