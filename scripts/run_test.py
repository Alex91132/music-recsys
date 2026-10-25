"""Финальная проверка на тестовой неделе 1498–1504 — запускается один раз.

Ранкер учится на окне VAL (история до 1491, метки 1491–1497) и применяется к окну TEST (история до 1498).
Всё, что выбиралось (источники, признаки, параметры), выбрано на валидации.

    python scripts/run_test.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import ranker as R        # noqa: E402
from recsys.data import TEST_START, VAL_START, history, load_events, mappings, targets   # noqa: E402
from recsys.metrics import item_popularity, paired_bootstrap, per_user, summary   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parents[1] / "results"
NAMES = {"own": "ещё раз своё", "artist": "исполнители", "i2i": "слушают вместе", "als": "ALS", "pop": "популярное"}


def main():
    ev = load_events()
    artists, _ = mappings()
    hist = history(ev, TEST_START)
    pop = item_popularity(hist)
    n_items = ev["item_id"].n_unique()
    act = hist.group_by("uid").agg(pl.len().alias("история"))
    rows, imps, peruser = [], [], []
    for task, discover in [("прослушает", False), ("откроет", True)]:
        train = R.build(ev, VAL_START, artists, discover)
        test = R.build(ev, TEST_START, artists, discover)
        tg = targets(ev, TEST_START, new_only=discover)
        model = R.fit(train)
        res = {"ранкер": R.predict(model, test)}
        for s in [c[2:] for c in test.data.columns if c.startswith("r_")]:
            res[NAMES[s]] = R.source_only(test, s)
        best = "ещё раз своё" if not discover else "ALS"
        pu = {n: per_user(r, tg, 10) for n, r in res.items()}
        for name, r in res.items():
            s = summary(r, tg, pop, n_items)
            d, lo, hi = paired_bootstrap(pu[best], pu[name], "ndcg@10")
            org = {}
            for part_name, part in [("ndcg@10 сам нашёл", tg.filter(pl.col("organic"))),
                                    ("ndcg@10 из рекомендаций", tg.filter(~pl.col("organic")))]:
                org[part_name] = round(float(per_user(r, part, 10)["ndcg@10"].mean()), 4)
            rows.append({"задача": task, "модель": name, **{k: round(v, 4) if isinstance(v, float) else v
                                                            for k, v in s.items()}, **org,
                         "лучший простой": best, "Δ ndcg@10": round(d, 4), "ДИ от": round(lo, 4), "ДИ до": round(hi, 4),
                         "полнота кандидатов": round(R.candidate_recall(test), 4)})
            print(rows[-1], flush=True)
        both = (pu["ранкер"].select("uid", pl.col("ndcg@10").alias("ранкер"))
                .join(pu[best].select("uid", pl.col("ndcg@10").alias("лучший простой")), on="uid")
                .join(act, on="uid", how="left").with_columns(pl.lit(task).alias("задача")))
        peruser.append(both)
        imps.append(R.importance(model).with_columns(pl.lit(task).alias("задача")))
    OUT.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_csv(OUT / "test.csv")
    pl.concat(imps).write_csv(OUT / "importance_test.csv")
    pl.concat(peruser).write_csv(OUT / "test_peruser.csv")


if __name__ == "__main__":
    main()
