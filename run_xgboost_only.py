# -*- coding: utf-8 -*-
"""
XGBoost 仅模型对比脚本（Source 训练 -> Target 测试）
数据目录示例：
/home/dbt8211210813/scAdaDrug/datasets/data/AR-42/Source_exprs_resp_z_AR-42.tsv
/home/dbt8211210813/scAdaDrug/datasets/data/AR-42/Target_exprs_resp_z_AR-42.tsv
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

import xgboost
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- 基本配置 ----------
DEFAULT_BASE = "/home/dbt8211210813/scAdaDrug/datasets/data"
DRUGS_ALL = [
    "AR-42",
    "Afatinib",
    "Cetuximab",
    "Etoposide",
    "Gefitinib",
    "NVP-TAE684",
    "PLX4720",
    "PLX4720_451Lu",
    "Sorafenib",
    "Vorinostat",
]
RANDOM_STATE = 42


# ---------- 工具函数 ----------
def safe_auc(y_true, proba1):
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, proba1)


def safe_aupr(y_true, proba1):
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return average_precision_score(y, proba1)


def compute_metrics(y_true, y_prob):
    """y_prob: (N,2) 或 (N,) 为正类概率"""
    y_true = np.asarray(y_true).astype(int)
    proba1 = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    y_pred = (proba1 >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = safe_auc(y_true, proba1)
    aupr = safe_aupr(y_true, proba1)
    return acc, auc, aupr


def infer_label_column(df: pd.DataFrame):
    for cand in ["response", "Response", "label", "Label", "y", "Y"]:
        if cand in df.columns:
            return cand
    return df.columns[-1]


def read_tsv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path)


def load_source_target(base_dir: str, drug: str):
    """
    返回：
      X_s, y_s, X_t, y_t, colnames
    操作：
      1) 读取 Source/Target
      2) 以列名取交集对齐
      3) 所有特征强制转 float，无法转换的用该列中位数填补
    """
    ddir = Path(base_dir) / drug
    src = ddir / f"Source_exprs_resp_z.{drug}.tsv"
    tgt = ddir / f"Target_expr_resp_z.{drug}.tsv"

    if not src.exists():
        raise FileNotFoundError(f"找不到文件：{src}")
    if not tgt.exists():
        raise FileNotFoundError(f"找不到文件：{tgt}")

    df_s = read_tsv(str(src))
    df_t = read_tsv(str(tgt))

    # 若有多余索引列，处理掉
    drop_unnamed = [c for c in df_s.columns if str(c).startswith("Unnamed")]
    if drop_unnamed:
        df_s = df_s.drop(columns=drop_unnamed, errors="ignore")
    drop_unnamed = [c for c in df_t.columns if str(c).startswith("Unnamed")]
    if drop_unnamed:
        df_t = df_t.drop(columns=drop_unnamed, errors="ignore")

    # 标签列
    ycol_s = infer_label_column(df_s)
    ycol_t = infer_label_column(df_t)

    y_s = df_s[ycol_s].astype(int).values
    y_t = df_t[ycol_t].astype(int).values

    Xs_df = df_s.drop(columns=[ycol_s])
    Xt_df = df_t.drop(columns=[ycol_t])

    # —— 对齐特征列（交集）——
    common_cols = sorted(list(set(Xs_df.columns).intersection(set(Xt_df.columns))))
    if len(common_cols) == 0:
        raise ValueError(f"{drug}: Source/Target 没有共同特征列，请检查表头")
    if len(common_cols) < Xs_df.shape[1] or len(common_cols) < Xt_df.shape[1]:
        print(f"[WARN] {drug}: 对齐特征列后保留 {len(common_cols)} 列 "
              f"(Source:{Xs_df.shape[1]} Target:{Xt_df.shape[1]})")
    Xs_df = Xs_df[common_cols].copy()
    Xt_df = Xt_df[common_cols].copy()

    # —— 强制转数值 & 填补 ——（关键修复）
    conv_cols = []
    fill_info = []

    for col in common_cols:
        # 先尝试一次是否全是数值型
        if not np.issubdtype(Xs_df[col].dtype, np.number) or not np.issubdtype(Xt_df[col].dtype, np.number):
            conv_cols.append(col)

        # 两边都转成 numeric；无法转的变成 NaN
        Xs_df[col] = pd.to_numeric(Xs_df[col], errors="coerce")
        Xt_df[col] = pd.to_numeric(Xt_df[col], errors="coerce")

        # 用 Source 该列的中位数填补（若全 NaN 则用 0）
        median = np.nanmedian(Xs_df[col].values)
        if np.isnan(median):
            median = 0.0

        n_nan_s = int(np.isnan(Xs_df[col].values).sum())
        n_nan_t = int(np.isnan(Xt_df[col].values).sum())

        if n_nan_s or n_nan_t:
            fill_info.append((col, n_nan_s, n_nan_t))

        Xs_df[col] = Xs_df[col].fillna(median)
        Xt_df[col] = Xt_df[col].fillna(median)

    if conv_cols:
        print(f"[WARN] {drug}: 发现非数值列（已转换为 float ）数量：{len(conv_cols)}")
    if fill_info:
        ex = "; ".join([f"{c}:S{ns}/T{nt}" for c, ns, nt in fill_info[:5]])
        if len(fill_info) > 5:
            ex += f"; ... 共 {len(fill_info)} 列发生填补"
        print(f"[WARN] {drug}: 存在无法直接转换为数值的条目，已用列中位数填补（示例）→ {ex}")

    return Xs_df.values, y_s, Xt_df.values, y_t, common_cols




def make_xgb(scale_pos_weight=None, eval_metric="aucpr"):
    """构建更稳健的 XGBoost；以 AUPR 为主优化目标。"""
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="auto",  # 有 GPU 可改 "gpu_hist"
        eval_metric=eval_metric,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
    )


def fit_with_early_stop(model, X_tr, y_tr, X_val, y_val, allow_early_stop=True):
    """
    尝试：
    1) sklearn API 的 early_stopping_rounds
    2) callbacks.EarlyStopping
    3) 无早停（兜底）
    """
    if not allow_early_stop:
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return "no_es"

    # 1) sklearn API
    try:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=50,
        )
        return "es_param"
    except TypeError:
        pass

    # 2) callbacks
    try:
        from xgboost.callback import EarlyStopping
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            callbacks=[EarlyStopping(rounds=50, metric_name="aucpr", save_best=True)],
        )
        return "es_callback"
    except Exception:
        # 3) 无早停
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return "no_es"


# ---------- 主流程 ----------
def run_one_drug(base_dir: str, drug: str, out_dir: str, no_early_stop=False):
    X_s, y_s, X_t, y_t, feat_names = load_source_target(base_dir, drug)

    # 不平衡：自动设置 scale_pos_weight
    pos = int((y_s == 1).sum())
    neg = int((y_s == 0).sum())
    spw = max(neg / max(pos, 1), 1.0) if pos > 0 and neg > 0 else None

    # Source 切出验证集用于早停
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_s, y_s, test_size=0.1, random_state=RANDOM_STATE, stratify=y_s
    )

    model = make_xgb(scale_pos_weight=spw, eval_metric="aucpr")
    es_mode = fit_with_early_stop(model, X_tr, y_tr, X_val, y_val, allow_early_stop=(not no_early_stop))

    # 在 Target 上评估
    prob_t = model.predict_proba(X_t)
    acc, auc, aupr = compute_metrics(y_t, prob_t)

    # 保存预测
    os.makedirs(out_dir, exist_ok=True)
    pred_df = pd.DataFrame({
        "drug": drug,
        "y_true": y_t.astype(int),
        "proba_0": prob_t[:, 0],
        "proba_1": prob_t[:, 1],
        "y_pred": (prob_t[:, 1] >= 0.5).astype(int),
    })
    pred_path = os.path.join(out_dir, f"pred_{drug}.csv")
    pred_df.to_csv(pred_path, index=False)

    return {
        "Drug": drug,
        "ACC": acc,
        "AUC": auc,
        "AUPR": aupr,
        "scale_pos_weight": spw,
        "early_stop_mode": es_mode,
        "best_ntree_limit": getattr(model, "best_ntree_limit", None),
        "pred_path": pred_path,
    }


def main():
    print("PY:", sys.executable)
    print("XGBoost:", xgboost.__version__, xgboost.__file__)

    parser = argparse.ArgumentParser(description="Run XGBoost (Source->Target) on scAdaDrug datasets")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE, help="数据根目录")
    parser.add_argument("--drug", type=str, default="ALL", help='单个药物名或 ALL')
    parser.add_argument("--out_dir", type=str, default="./xgb_results", help="输出目录")
    parser.add_argument("--no_early_stop", action="store_true", help="强制关闭早停")
    args = parser.parse_args()

    drugs = DRUGS_ALL if args.drug.upper() == "ALL" else [args.drug]

    results = []
    for d in drugs:
        print(f"\n===== Running {d} =====")
        try:
            info = run_one_drug(args.base_dir, d, args.out_dir, no_early_stop=args.no_early_stop)
            print(f"[{d}] ACC={info['ACC']:.4f}  AUC={info['AUC']:.4f}  AUPR={info['AUPR']:.4f}  "
                  f"spw={info['scale_pos_weight']}  es={info['early_stop_mode']}")
            results.append(info)
        except Exception as e:
            print(f"[ERROR] {d}: {e}")

    if results:
        df = pd.DataFrame(results)
        for c in ["ACC", "AUC", "AUPR"]:
            df[c] = df[c].round(3)
        df_avg = df[["ACC", "AUC", "AUPR"]].mean().round(3)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(args.out_dir, f"xgb_summary_{ts}.csv")
        df.to_csv(out_csv, index=False)

        print("\n===== Summary by Drug =====")
        print(df[["Drug", "ACC", "AUC", "AUPR", "early_stop_mode"]])
        print("\n===== Macro Average =====")
        print(df_avg.to_frame(name="mean").T)
        print(f"\n结果已保存到：{out_csv}")
    else:
        print("没有得到有效结果，请检查数据路径或文件名。")


if __name__ == "__main__":
    main()
