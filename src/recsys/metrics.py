"""Метрики ранжирования по пользователям и бутстреп-интервалы.

Выдача — таблица (uid, item_id, rank), rank начинается с 1. Цели — (uid, item_id[, organic]).
Пользователи без выдачи получают нули (а не выпадают из среднего).
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl


def per_user(recs: pl.DataFrame, targets: pl.DataFrame, k: int = 10) -> pl.DataFrame:
    """uid, n_true, hits@k, recall@k, ndcg@k для каждого пользователя из targets."""
    n_true = targets.group_by("uid").agg(pl.len().alias("n_true"))
    hit = (recs.filter(pl.col("rank") <= k)
           .join(targets.select("uid", "item_id"), on=["uid", "item_id"], how="inner")
           .group_by("uid")
           .agg(pl.len().alias("hits"), (1 / (pl.col("rank") + 1).log(2)).sum().alias("dcg")))
    idcg = {m: sum(1 / math.log2(i + 1) for i in range(1, m + 1)) for m in range(1, k + 1)}
    out = (n_true.join(hit, on="uid", how="left").fill_null(0)
           .with_columns(pl.col("n_true").clip(upper_bound=k).replace_strict(idcg, return_dtype=pl.Float64).alias("idcg")))
    return out.select("uid", "n_true", "hits",
                      (pl.col("hits") / pl.col("n_true")).alias(f"recall@{k}"),
                      (pl.col("dcg") / pl.col("idcg")).alias(f"ndcg@{k}"))


def summary(recs: pl.DataFrame, targets: pl.DataFrame, item_pop: pl.DataFrame, n_items: int,
            ks=(10, 100)) -> dict:
    """Средние по пользователям + покрытие каталога, новизна и доля «головы» в топ-10."""
    out = {}
    for k in ks:
        u = per_user(recs, targets, k)
        out[f"ndcg@{k}"] = float(u[f"ndcg@{k}"].mean())
        out[f"recall@{k}"] = float(u[f"recall@{k}"].mean())
    top = recs.filter(pl.col("rank") <= 10)
    out["coverage@10"] = top["item_id"].n_unique() / n_items
    pop = item_pop.select("item_id", "share", "head")
    t = top.join(pop, on="item_id", how="left").fill_null(0)
    out["novelty@10"] = float((-(t["share"].clip(lower_bound=1e-9)).log(2)).mean())
    out["head@10"] = float(t["head"].cast(pl.Float64).mean())
    out["users"] = targets["uid"].n_unique()
    return out


def item_popularity(hist: pl.DataFrame) -> pl.DataFrame:
    """Доля положительных откликов на трек в истории; head — топ-1 % треков по популярности."""
    p = hist.filter(pl.col("pos")).group_by("item_id").agg(pl.len().alias("n"))
    p = p.with_columns((pl.col("n") / pl.col("n").sum()).alias("share"))
    thr = p["n"].quantile(0.99)
    return p.with_columns((pl.col("n") >= thr).alias("head"))


def paired_bootstrap(a: pl.DataFrame, b: pl.DataFrame, col: str, n_boot: int = 2000, seed: int = 0):
    """Разница средних (b − a) по одним и тем же пользователям и 95 % ДИ бутстрепом по пользователям."""
    m = a.select("uid", pl.col(col).alias("a")).join(b.select("uid", pl.col(col).alias("b")), on="uid")
    d = (m["b"] - m["a"]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
