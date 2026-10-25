"""Описательная статистика для отчёта → results/eda_*.csv (маленькие агрегаты, данные в git не нужны).

    python scripts/eda.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.data import TEST_START, history, load_events   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    ev = history(load_events(), TEST_START)          # тестовую неделю в описании не используем
    raw = pl.read_parquet(Path(__file__).resolve().parents[1] / "data" / "raw" / "multi_event.parquet",
                          columns=["event_type", "played_ratio_pct", "is_organic"])
    facts = {
        "событий": ev.height, "пользователей": ev["uid"].n_unique(), "треков": ev["item_id"].n_unique(),
        "дней": int(ev["day"].max()) + 1,
        "доля из рекомендаций": float(1 - ev.filter(pl.col("listen"))["organic"].mean()),
        "доля listen+": float(ev.filter(pl.col("listen"))["pos"].mean()),
        "доля пропусков (≤10 %)": float(ev.filter(pl.col("listen"))["skip"].mean()),
    }
    L = ev.filter(pl.col("listen"))
    facts["доля повторных прослушиваний"] = 1 - L.select(pl.struct("uid", "item_id").n_unique()).item() / L.height
    pl.DataFrame([facts]).write_csv(OUT / "eda_facts.csv")

    # длинный хвост: доля положительных откликов у топ-x % треков
    n = ev.filter(pl.col("pos")).group_by("item_id").agg(pl.len().alias("n")).sort("n", descending=True)
    c = n["n"].cum_sum() / n["n"].sum()
    xs = np.unique(np.r_[np.geomspace(1, n.height, 200).astype(int), n.height]) - 1
    pl.DataFrame({"доля треков": (xs + 1) / n.height, "доля откликов": c.to_numpy()[xs]}).write_csv(OUT / "eda_longtail.csv")

    # активность пользователей
    u = ev.group_by("uid").agg(pl.len().alias("событий"), pl.col("organic").mean().alias("доля органики"),
                               pl.col("skip").mean().alias("доля пропусков"))
    u.write_csv(OUT / "eda_users.csv")

    # распределение доли прослушанного, по источнику
    pr = (raw.filter(pl.col("event_type") == "listen")
          .with_columns(pl.col("played_ratio_pct").clip(upper_bound=110).alias("pct"))
          .group_by("pct", "is_organic").agg(pl.len().alias("n")).sort("pct"))
    pr.write_csv(OUT / "eda_played.csv")

    # недельная динамика: активные пользователи и доля органики
    w = (ev.with_columns((pl.col("day") // 7).alias("неделя"))
         .group_by("неделя").agg(pl.col("uid").n_unique().alias("активных"), pl.len().alias("событий"),
                                 pl.col("organic").mean().alias("доля органики")).sort("неделя"))
    w.write_csv(OUT / "eda_weekly.csv")
    print(facts)


if __name__ == "__main__":
    main()
