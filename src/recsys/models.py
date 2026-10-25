"""Источники кандидатов посложнее: совместные прослушивания (item2item) и ALS.

Обе модели учатся только на истории (события до начала окна) и возвращают выдачу (uid, item_id, rank, score).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp

from .baselines import _exclude, _rank, decay


def _decay(end: int, half_life: float) -> pl.Expr:
    return decay(end, half_life)


def item2item(hist: pl.DataFrame, users: pl.Series, k: int, days: int = 180, window: int = 5,
              n_seed: int = 50, half_life: float = 30.0, exclude: pl.DataFrame | None = None) -> pl.DataFrame:
    """«Слушают вместе»: пары треков, положительно прослушанных подряд (до window шагов) одним пользователем.

    Сходство — косинус по числу совместных появлений: c_ij / sqrt(n_i · n_j). Оценка трека для пользователя —
    сумма сходств с его последними n_seed положительно прослушанными треками, взвешенных по давности.
    """
    end = int(hist["day"].max())
    h = (hist.filter(pl.col("pos") & (pl.col("day") > end - days))
         .sort("uid", "ts").select("uid", "item_id", "day"))
    pairs = []
    for lag in range(1, window + 1):
        nxt = h.with_columns(pl.col("item_id").shift(-lag).over("uid").alias("j"))
        pairs.append(nxt.filter(pl.col("j").is_not_null() & (pl.col("j") != pl.col("item_id")))
                     .select(pl.col("item_id").alias("i"), "j"))
    p = pl.concat(pairs)
    p = pl.concat([p, p.select(pl.col("j").alias("i"), pl.col("i").alias("j"))])     # симметрия
    c = p.group_by("i", "j").agg(pl.len().alias("c")).filter(pl.col("c") >= 2)
    n = h.group_by("item_id").agg(pl.len().alias("n"))
    c = (c.join(n.rename({"item_id": "i", "n": "ni"}), on="i")
         .join(n.rename({"item_id": "j", "n": "nj"}), on="j")
         .with_columns((pl.col("c") / (pl.col("ni") * pl.col("nj")).sqrt()).alias("sim")))
    c = c.filter(pl.col("sim").rank("ordinal", descending=True).over("i") <= 100).select("i", "j", "sim")
    seeds = (h.filter(pl.col("uid").is_in(users))
             .group_by("uid", "item_id").agg(_decay(end, half_life).sum().alias("w"))
             .filter(pl.col("w").rank("ordinal", descending=True).over("uid") <= n_seed))
    cand = (seeds.join(c, left_on="item_id", right_on="i")
            .group_by("uid", pl.col("j").alias("item_id")).agg((pl.col("w") * pl.col("sim")).sum().alias("score")))
    return _rank(_exclude(cand, exclude), "score", k)


def als(hist: pl.DataFrame, users: pl.Series, k: int, factors: int = 128, iterations: int = 15,
        regularization: float = 0.05, alpha: float = 20.0, days: int = 365, half_life: float = 90.0,
        min_item: int = 3, exclude: pl.DataFrame | None = None, seed: int = 0) -> pl.DataFrame:
    """ALS (implicit) на матрице «пользователь × трек» с весом = затухающая сумма положительных откликов."""
    from implicit.als import AlternatingLeastSquares

    end = int(hist["day"].max())
    h = hist.filter(pl.col("pos") & (pl.col("day") > end - days))
    w = h.group_by("uid", "item_id").agg(_decay(end, half_life).sum().alias("w"))
    keep = w.group_by("item_id").agg(pl.len().alias("n")).filter(pl.col("n") >= min_item)
    w = w.join(keep.select("item_id"), on="item_id")
    uids = w["uid"].unique().sort()
    iids = w["item_id"].unique().sort()
    umap = pl.DataFrame({"uid": uids, "u": np.arange(len(uids), dtype=np.int32)})
    imap = pl.DataFrame({"item_id": iids, "i": np.arange(len(iids), dtype=np.int32)})
    w = w.join(umap, on="uid").join(imap, on="item_id")
    mat = sp.csr_matrix(((1 + alpha * w["w"]).to_numpy().astype(np.float32),
                         (w["u"].to_numpy(), w["i"].to_numpy())), shape=(len(uids), len(iids)))
    model = AlternatingLeastSquares(factors=factors, iterations=iterations, regularization=regularization,
                                    random_state=seed, calculate_training_loss=False)
    model.fit(mat, show_progress=False)
    target = umap.join(pl.DataFrame({"uid": users}), on="uid")
    n_ask = k + (500 if exclude is not None else 0)
    ids, scores = model.recommend(target["u"].to_numpy(), mat[target["u"].to_numpy()], N=n_ask,
                                  filter_already_liked_items=exclude is not None)
    cand = pl.DataFrame({"uid": np.repeat(target["uid"].to_numpy(), n_ask),
                         "item_id": iids.to_numpy()[ids.ravel()], "score": scores.ravel()})
    cand = cand.filter(pl.col("score") > -1e30)
    return _rank(_exclude(cand, exclude), "score", k)
