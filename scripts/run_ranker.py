"""Ранкер на валидации: учим на окне DEV (метки 1484–1490), проверяем на VAL (1491–1497). Тест не трогаем.

    python scripts/run_ranker.py
"""
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import ranker as R        # noqa: E402
from recsys.data import DEV_START, VAL_START, history, load_events, mappings, targets   # noqa: E402
from recsys.metrics import item_popularity, paired_bootstrap, per_user, summary   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    ev = load_events()
    artists, _ = mappings()
    n_items = ev["item_id"].n_unique()
    pop = item_popularity(history(ev, VAL_START))
    rows, imps = [], []
    for task, discover in [("прослушает", False), ("откроет", True)]:
        t = time.time()
        dev = R.build(ev, DEV_START, artists, discover)
        val = R.build(ev, VAL_START, artists, discover)
        print(task, "кандидатов на пользователя:", round(val.data.height / val.data["uid"].n_unique()),
              "| полнота кандидатов:", round(R.candidate_recall(val), 4), f"| {time.time() - t:.0f} с", flush=True)
        model = R.fit(dev)
        tg = targets(ev, VAL_START, new_only=discover)
        best_src = "own" if not discover else "als"
        res = {"ранкер": R.predict(model, val), **{f"только {s}": R.source_only(val, s)
                                                   for s in (["own", "als", "artist", "i2i"] if not discover
                                                             else ["als", "artist", "i2i"])}}
        pu = {name: per_user(r, tg, 10) for name, r in res.items()}
        for name, r in res.items():
            s = summary(r, tg, pop, n_items)
            d, lo, hi = paired_bootstrap(pu[f"только {best_src}"], pu[name], "ndcg@10")
            rows.append({"задача": task, "модель": name, **{k: round(v, 4) if isinstance(v, float) else v
                                                            for k, v in s.items()},
                         f"Δ ndcg@10 к {best_src}": round(d, 4), "ДИ от": round(lo, 4), "ДИ до": round(hi, 4)})
            print(rows[-1], flush=True)
        imps.append(R.importance(model).with_columns(pl.lit(task).alias("задача")))
    OUT.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_csv(OUT / "ranker_val.csv")
    pl.concat(imps).write_csv(OUT / "ranker_importance_val.csv")
    print(pl.concat(imps).group_by("задача").head(8))


if __name__ == "__main__":
    main()
