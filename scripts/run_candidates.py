"""Все источники кандидатов на валидационной неделе: качество ранжирования и полнота кандидатов (Recall@200).

    python scripts/run_candidates.py [модель ...]
"""
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import baselines as B        # noqa: E402
from recsys import models as M           # noqa: E402
from recsys.data import VAL_START, history, load_events, mappings, seen_items, targets   # noqa: E402
from recsys.metrics import item_popularity, per_user, summary   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parents[1] / "results"
K = 200


def main(only: list[str]):
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
            "исполнители пользователя": lambda: B.artist_tracks(hist, users, artists, K, exclude=ex),
            "слушают вместе (item2item)": lambda: M.item2item(hist, users, K, exclude=ex),
            "ALS": lambda: M.als(hist, users, K, exclude=ex),
        }
        if not new_only:
            models["ещё раз своё"] = lambda: B.own_top(hist, users, K)
        for name, fn in models.items():
            if only and not any(o.lower() in name.lower() for o in only):
                continue
            t = time.time()
            r = fn()
            s = summary(r, tg, pop, n_items)
            s["recall@200"] = float(per_user(r, tg, 200)["recall@200"].mean())
            rows.append({"задача": task, "модель": name, **{k: round(v, 4) if isinstance(v, float) else v
                                                            for k, v in s.items()}, "сек": round(time.time() - t, 1)})
            print(rows[-1], flush=True)
    OUT.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_csv(OUT / ("candidates_val.csv" if not only else "candidates_val_partial.csv"))


if __name__ == "__main__":
    main(sys.argv[1:])
