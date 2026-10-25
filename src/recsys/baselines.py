"""Простые модели, которые нужно побить: популярное, «ещё раз своё», треки любимых исполнителей.

Каждая функция возвращает выдачу (uid, item_id, rank) для заданных пользователей. exclude — пары
(uid, item_id), которые нельзя рекомендовать (в задаче «откроет» — всё, что пользователь уже встречал).
"""
from __future__ import annotations

import polars as pl


def _rank(df: pl.DataFrame, score: str, k: int) -> pl.DataFrame:
    return (df.sort(["uid", score], descending=[False, True])
            .with_columns(pl.col(score).rank("ordinal", descending=True).over("uid").alias("rank"))
            .filter(pl.col("rank") <= k).select("uid", "item_id", pl.col("rank").cast(pl.UInt32)))


def _exclude(cand: pl.DataFrame, exclude: pl.DataFrame | None) -> pl.DataFrame:
    return cand if exclude is None else cand.join(exclude, on=["uid", "item_id"], how="anti")


def popular(hist: pl.DataFrame, users: pl.Series, k: int, days: int | None = 30,
            exclude: pl.DataFrame | None = None) -> pl.DataFrame:
    """Одинаковый для всех топ по положительным откликам (за последние days дней или за всё время)."""
    h = hist.filter(pl.col("pos"))
    if days is not None:
        h = h.filter(pl.col("day") >= hist["day"].max() - days + 1)
    top = h.group_by("item_id").agg(pl.len().alias("score")).sort("score", descending=True)
    pool = top.head(k + (5_000 if exclude is not None else 0))      # запас под исключение уже знакомого
    cand = pl.DataFrame({"uid": users}).join(pool, how="cross")
    return _rank(_exclude(cand, exclude), "score", k)


def own_top(hist: pl.DataFrame, users: pl.Series, k: int, half_life: float = 60.0) -> pl.DataFrame:
    """«Ещё раз своё»: треки пользователя по числу положительных откликов с затуханием по давности."""
    end = hist["day"].max()
    h = hist.filter(pl.col("pos") & pl.col("uid").is_in(users))
    s = h.group_by("uid", "item_id").agg(
        (0.5 ** ((end - pl.col("day").cast(pl.Float64)) / half_life)).sum().alias("score"))
    return _rank(s, "score", k)


def artist_tracks(hist: pl.DataFrame, users: pl.Series, artists: pl.DataFrame, k: int, n_artists: int = 30,
                  half_life: float = 60.0, exclude: pl.DataFrame | None = None) -> pl.DataFrame:
    """Популярные треки любимых исполнителей пользователя.

    Вес исполнителя — затухающая сумма положительных откликов пользователя на его треки;
    вес трека внутри исполнителя — доля положительных откликов всех пользователей за 90 дней.
    """
    end = hist["day"].max()
    h = hist.filter(pl.col("pos")).join(artists, on="item_id", how="inner")
    ua = (h.filter(pl.col("uid").is_in(users))
          .group_by("uid", "artist_id")
          .agg((0.5 ** ((end - pl.col("day").cast(pl.Float64)) / half_life)).sum().alias("w_artist"))
          .with_columns(pl.col("w_artist").rank("ordinal", descending=True).over("uid").alias("r"))
          .filter(pl.col("r") <= n_artists))
    ua = ua.with_columns(pl.col("w_artist") / pl.col("w_artist").sum().over("uid"))
    recent = h.filter(pl.col("day") >= end - 89)
    it = recent.group_by("artist_id", "item_id").agg(pl.len().alias("n"))
    it = it.with_columns((pl.col("n") / pl.col("n").sum().over("artist_id")).alias("w_item"))
    it = it.filter(pl.col("n").rank("ordinal", descending=True).over("artist_id") <= 50)
    cand = ua.join(it, on="artist_id").with_columns((pl.col("w_artist") * pl.col("w_item")).alias("score"))
    cand = cand.group_by("uid", "item_id").agg(pl.col("score").sum())
    return _rank(_exclude(cand, exclude), "score", k)
