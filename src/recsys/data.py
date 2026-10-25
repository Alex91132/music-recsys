"""События Yambda-50M → компактная таблица с днём и типом отклика; окна «история / неделя-цель».

Время в датасете — число 5-секундных интервалов от начала; день = timestamp · 5 // 86 400.
Разбиение (см. PLAN.md): история до начала окна, цель — 7 дней с начала окна.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "processed" / "events.parquet"

WEEK = 7
TEST_START = 1498      # тест: 1498–1504, открывается один раз
VAL_START = 1491       # валидация: 1491–1497
DEV_START = 1484       # окно для обучения ранкера при подборе на валидации
LISTEN_PLUS = 50       # прослушано больше 50 % — положительный отклик (определение авторов)
SKIP = 10              # прослушано не больше 10 % — пропуск


def load_events(rebuild: bool = False) -> pl.DataFrame:
    """uid, item_id, ts, day, organic, pos (listen+ или лайк), listen, skip, like, dislike."""
    if CACHE.exists() and not rebuild:
        return pl.read_parquet(CACHE)
    e = pl.read_parquet(RAW / "multi_event.parquet")
    t = pl.col("event_type")
    out = e.select(
        "uid", "item_id",
        pl.col("timestamp").alias("ts"),
        (pl.col("timestamp").cast(pl.UInt64) * 5 // 86_400).cast(pl.UInt16).alias("day"),
        pl.col("is_organic").cast(pl.Boolean).alias("organic"),
        (t == "listen").alias("listen"),
        ((t == "listen") & (pl.col("played_ratio_pct") > LISTEN_PLUS) | (t == "like")).alias("pos"),
        ((t == "listen") & (pl.col("played_ratio_pct") <= SKIP)).alias("skip"),
        (t == "like").alias("like"),
        (t == "dislike").alias("dislike"),
    )
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(CACHE)
    return out


def history(events: pl.DataFrame, start: int) -> pl.DataFrame:
    return events.filter(pl.col("day") < start)


def targets(events: pl.DataFrame, start: int, new_only: bool = False, days: int = WEEK) -> pl.DataFrame:
    """Положительные отклики (uid, item_id, organic) в окне [start, start + days).

    new_only — только треки, с которыми у пользователя не было ни одного события до start («откроет»).
    organic — был ли хотя бы один из откликов на трек органическим.
    """
    w = events.filter((pl.col("day") >= start) & (pl.col("day") < start + days) & pl.col("pos"))
    t = w.group_by("uid", "item_id").agg(pl.col("organic").any())
    if new_only:
        seen = history(events, start).select("uid", "item_id").unique()
        t = t.join(seen, on=["uid", "item_id"], how="anti")
    return t


def seen_items(events: pl.DataFrame, start: int) -> pl.DataFrame:
    """Все пары (uid, item_id) с любым событием до start — исключаются из выдачи в задаче «откроет»."""
    return history(events, start).select("uid", "item_id").unique()


def mappings() -> tuple[pl.DataFrame, pl.DataFrame]:
    return (pl.read_parquet(RAW / "artist_item_mapping.parquet"),
            pl.read_parquet(RAW / "album_item_mapping.parquet"))
