"""Ловушка: случайное разбиение вместо разбиения по времени завышает качество.

Одна и та же модель (ALS с теми же параметрами), одинаковый объём проверочных данных — неделя откликов:
  • по времени: учимся на всём до дня 1491, проверяем на неделе 1491–1497 (как в проекте);
  • случайно: из последних 8 недель (1441–1497) случайно откладываем 1/8 событий, учимся на остальном
    (включая будущие недели), проверяем на отложенном.

    python scripts/run_leakage.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import models as M        # noqa: E402
from recsys.data import VAL_START, WEEK, load_events   # noqa: E402
from recsys.metrics import per_user   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parents[1] / "results"
K = 100


def evaluate(train: pl.DataFrame, test: pl.DataFrame, discover: bool):
    tg = test.filter(pl.col("pos")).select("uid", "item_id").unique()
    seen = train.select("uid", "item_id").unique()
    if discover:
        tg = tg.join(seen, on=["uid", "item_id"], how="anti")
    users = tg["uid"].unique()
    recs = M.als(train, users, K, exclude=seen if discover else None)
    return per_user(recs, tg, 10).join(per_user(recs, tg, K).select("uid", f"recall@{K}"), on="uid"), tg.height


def main():
    ev = load_events().filter(pl.col("day") < VAL_START + WEEK)
    rng = np.random.default_rng(0)
    pool = ev["day"] >= VAL_START - 7 * WEEK
    hold = pool & pl.Series(rng.random(ev.height) < 1 / 8)
    rows = []
    for task, discover in [("прослушает", False), ("откроет", True)]:
        res = {}
        for split, (tr, te) in {"по времени": (ev.filter(pl.col("day") < VAL_START), ev.filter(pl.col("day") >= VAL_START)),
                                "случайно": (ev.filter(~hold), ev.filter(hold))}.items():
            pu, n = evaluate(tr, te, discover)
            res[split] = pu
            rows.append({"задача": task, "разбиение": split, "ndcg@10": round(float(pu["ndcg@10"].mean()), 4),
                         f"recall@{K}": round(float(pu[f"recall@{K}"].mean()), 4), "пользователей": pu.height,
                         "откликов в проверке": n, "завышение, раз": None})
            print(rows[-1], flush=True)
        r_t = rows[-2]["ndcg@10"]
        rows[-1]["завышение, раз"] = round(rows[-1]["ndcg@10"] / r_t, 1) if r_t else None
    OUT.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_csv(OUT / "leakage_val.csv")
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
