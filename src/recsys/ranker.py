"""Двухэтапная схема: кандидаты из всех источников → признаки → LightGBM LambdaRank.

Окно задаётся началом недели-цели start: история — всё до start, метки — отклики в [start, start + 7).
Ранкер учится на одном окне (метки известны) и применяется к следующему: признаки для обоих
считаются только по их истории, так что будущее в признаки не попадает.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from . import baselines as B
from .baselines import decay
from . import models as M
from .data import history, seen_items, targets

N_CAND = {"own": 150, "artist": 150, "i2i": 150, "als": 200, "pop": 50}


def _decay(end: int, half_life: float) -> pl.Expr:
    return decay(end, half_life)


def candidates(hist: pl.DataFrame, users: pl.Series, artists: pl.DataFrame, discover: bool,
               exclude: pl.DataFrame | None) -> pl.DataFrame:
    """Объединение источников: (uid, item_id) + ранг трека в каждом источнике (null — источник его не предложил)."""
    src = {"artist": B.artist_tracks(hist, users, artists, N_CAND["artist"], exclude=exclude),
           "i2i": M.item2item(hist, users, N_CAND["i2i"], exclude=exclude),
           "als": M.als(hist, users, N_CAND["als"], exclude=exclude),
           "pop": B.popular(hist, users, N_CAND["pop"], 30, exclude)}
    if not discover:
        src["own"] = B.own_top(hist, users, N_CAND["own"])
    out = None
    for name, r in src.items():
        r = r.select("uid", "item_id", pl.col("rank").cast(pl.Float32).alias(f"r_{name}"))
        out = r if out is None else out.join(r, on=["uid", "item_id"], how="full", coalesce=True)
    return out


def features(cand: pl.DataFrame, hist: pl.DataFrame, artists: pl.DataFrame) -> pl.DataFrame:
    """Признаки трека, пользователя и пары — только из истории."""
    end = int(hist["day"].max())
    L = hist.filter(pl.col("listen"))
    it = hist.group_by("item_id").agg(
        pl.col("pos").filter(pl.col("day") > end - 7).sum().alias("i_pos7"),
        pl.col("pos").filter(pl.col("day") > end - 30).sum().alias("i_pos30"),
        pl.col("pos").sum().alias("i_pos_all"),
        pl.col("uid").filter(pl.col("day") > end - 30).n_unique().alias("i_users30"),
        pl.col("like").sum().alias("i_likes"),
        pl.col("dislike").sum().alias("i_dislikes"),
        pl.col("day").min().alias("i_first_day"),
        pl.col("organic").mean().alias("i_organic"),
    )
    il = L.group_by("item_id").agg(pl.col("skip").mean().alias("i_skip_rate"), pl.len().alias("i_listens"))
    it = (it.join(il, on="item_id", how="left")
          .with_columns((end - pl.col("i_first_day").cast(pl.Int32)).alias("i_age"),
                        (pl.col("i_pos7") / (pl.col("i_pos30") + 1)).alias("i_trend"),
                        (pl.col("i_likes") / (pl.col("i_listens") + 1)).alias("i_like_rate"))
          .drop("i_first_day"))
    us = hist.group_by("uid").agg(
        pl.len().alias("u_events"),
        pl.col("pos").filter(pl.col("day") > end - 30).sum().alias("u_pos30"),
        pl.col("organic").mean().alias("u_organic"),
        pl.col("skip").mean().alias("u_skip_rate"),
        pl.col("item_id").n_unique().alias("u_items"),
        (end - pl.col("day").max().cast(pl.Int32)).alias("u_days_since"),
    )
    ui = hist.filter(pl.col("uid").is_in(cand["uid"].unique())).group_by("uid", "item_id").agg(
        pl.col("pos").sum().alias("ui_pos"), pl.col("skip").sum().alias("ui_skips"),
        (end - pl.col("day").max().cast(pl.Int32)).alias("ui_days_since"))
    a = artists.unique("item_id")
    ua = (hist.filter(pl.col("pos") & pl.col("uid").is_in(cand["uid"].unique()))
          .join(a, on="item_id").group_by("uid", "artist_id")
          .agg(_decay(end, 60).sum().alias("ua_w"), (end - pl.col("day").max().cast(pl.Int32)).alias("ua_days_since"))
          .with_columns((pl.col("ua_w") / pl.col("ua_w").sum().over("uid")).alias("ua_share")))
    f = (cand.join(it, on="item_id", how="left").join(us, on="uid", how="left")
         .join(ui, on=["uid", "item_id"], how="left")
         .join(a, on="item_id", how="left").join(ua, on=["uid", "artist_id"], how="left").drop("artist_id"))
    return f


@dataclass
class Window:
    start: int
    discover: bool
    data: pl.DataFrame          # кандидаты + признаки + label
    n_pos_total: int            # всего положительных в окне (для полноты кандидатов)


def build(events: pl.DataFrame, start: int, artists: pl.DataFrame, discover: bool, cache: bool = True) -> Window:
    """Кандидаты + признаки + метки окна; результат кэшируется в data/processed (пересчёт ~1,5 мин)."""
    from .data import ROOT
    tg = targets(events, start, new_only=discover)
    path = ROOT / "data" / "processed" / f"window_{start}_{'new' if discover else 'all'}.parquet"
    if cache and path.exists():
        return Window(start, discover, pl.read_parquet(path), tg.height)
    hist = history(events, start)
    users = tg["uid"].unique()
    ex = seen_items(events, start) if discover else None
    cand = candidates(hist, users, artists, discover, ex)
    f = features(cand, hist, artists)
    f = f.join(tg.select("uid", "item_id", pl.lit(1, pl.Int8).alias("label")), on=["uid", "item_id"], how="left")
    f = f.with_columns(pl.col("label").fill_null(0)).sort("uid")
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        f.write_parquet(path)
    return Window(start, discover, f, tg.height)


def without_source(win: Window, src: str) -> Window:
    """Окно без одного источника: убираем его признак-ранг и кандидатов, которых предложил только он."""
    others = [c for c in win.data.columns if c.startswith("r_") and c != f"r_{src}"]
    keep = pl.any_horizontal([pl.col(c).is_not_null() for c in others])
    return Window(win.start, win.discover, win.data.filter(keep).drop(f"r_{src}"), win.n_pos_total)


FEATURES_EXCLUDE = {"uid", "item_id", "label"}


def fit(train: Window, seed: int = 0, n_estimators: int = 400) -> lgb.LGBMRanker:
    d = train.data.filter(pl.col("label").sum().over("uid") > 0)        # группы без положительных не учат ранжированию
    cols = [c for c in d.columns if c not in FEATURES_EXCLUDE]
    grp = d.group_by("uid", maintain_order=True).len()["len"].to_numpy()
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=n_estimators, learning_rate=0.05, num_leaves=63,
                           min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           lambdarank_truncation_level=20, random_state=seed, verbose=-1)
    model.fit(d.select(cols).to_pandas(), d["label"].to_numpy(), group=grp)
    model.feature_names_ = cols
    return model


def predict(model: lgb.LGBMRanker, win: Window, k: int = 100) -> pl.DataFrame:
    d = win.data
    s = model.predict(d.select(model.feature_names_).to_pandas())
    r = d.select("uid", "item_id").with_columns(pl.Series("score", s))
    return (r.sort(["uid", "score"], descending=[False, True])
            .with_columns(pl.col("score").rank("ordinal", descending=True).over("uid").alias("rank"))
            .filter(pl.col("rank") <= k).select("uid", "item_id", pl.col("rank").cast(pl.UInt32)))


def importance(model: lgb.LGBMRanker) -> pl.DataFrame:
    g = model.booster_.feature_importance("gain")
    return pl.DataFrame({"feature": model.feature_names_, "gain": g / g.sum()}).sort("gain", descending=True)


def candidate_recall(win: Window) -> float:
    return float(win.data["label"].sum()) / max(win.n_pos_total, 1)


def source_only(win: Window, src: str, k: int = 100) -> pl.DataFrame:
    """Выдача одного источника из того же набора кандидатов — для честного сравнения с ранкером."""
    d = win.data.filter(pl.col(f"r_{src}").is_not_null())
    return (d.select("uid", "item_id", pl.col(f"r_{src}").alias("score"))
            .with_columns(pl.col("score").rank("ordinal").over("uid").alias("rank"))
            .filter(pl.col("rank") <= k).select("uid", "item_id", pl.col("rank").cast(pl.UInt32)))


def _np(x):
    return np.asarray(x)
