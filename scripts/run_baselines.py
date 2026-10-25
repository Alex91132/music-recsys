"""Простые модели на валидационной неделе (тест не трогаем).

    python scripts/run_baselines.py
"""
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import baselines as B        # noqa: E402
from recsys.data import VAL_START, history, load_events, mappings, seen_items, targets   # noqa: E402
from recsys.metrics import item_popularity, summary   # noqa: E402

K = 100
sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    t0 = time.time()
    ev = load_events()
    hist = history(ev, VAL_START)
    artists, _ = mappings()
    pop = item_popularity(hist)
    n_items = ev["item_id"].n_unique()
    seen = seen_items(ev, VAL_START)
    rows = []
    for task, new_only in [("прослушает", False), ("откроет", True)]:
        tg = targets(ev, VAL_START, new_only=new_only)
        users = tg["uid"].unique()
        ex = seen if new_only else None
        models = {
            "популярное 30 дней": lambda: B.popular(hist, users, K, 30, ex),
            "популярное всё время": lambda: B.popular(hist, users, K, None, ex),
            "исполнители пользователя": lambda: B.artist_tracks(hist, users, artists, K, exclude=ex),
        }
        if not new_only:
            models["ещё раз своё"] = lambda: B.own_top(hist, users, K)
        for name, fn in models.items():
            t = time.time()
            r = fn()
            s = summary(r, tg, pop, n_items)
            rows.append({"задача": task, "модель": name, **{k: round(v, 4) if isinstance(v, float) else v
                                                            for k, v in s.items()}, "сек": round(time.time() - t, 1)})
            print(rows[-1], flush=True)
    OUT.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_csv(OUT / "baselines_val.csv")
    print(pl.DataFrame(rows))
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
