"""Анализ на валидации: смещение обратной связи (organic) и вклад источников кандидатов (абляция).

    python scripts/run_analysis_val.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import ranker as R        # noqa: E402
from recsys.data import DEV_START, VAL_START, load_events, mappings, targets   # noqa: E402
from recsys.metrics import paired_bootstrap, per_user   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parents[1] / "results"
SOURCES = {"own": "ещё раз своё", "artist": "исполнители", "i2i": "слушают вместе", "als": "ALS", "pop": "популярное"}


def organic_split(recs, tg):
    """NDCG@10 отдельно по откликам на треки, найденные самостоятельно, и пришедшие из рекомендаций."""
    out = {}
    for name, part in [("сам нашёл", tg.filter(pl.col("organic"))), ("из рекомендаций", tg.filter(~pl.col("organic")))]:
        out[name] = float(per_user(recs, part, 10)["ndcg@10"].mean())
        out[f"n {name}"] = part["uid"].n_unique()
    return out


def main():
    ev = load_events()
    artists, _ = mappings()
    bias, abl = [], []
    for task, discover in [("прослушает", False), ("откроет", True)]:
        dev = R.build(ev, DEV_START, artists, discover)
        val = R.build(ev, VAL_START, artists, discover)
        tg = targets(ev, VAL_START, new_only=discover)
        full = R.fit(dev)
        recs = {"ранкер": R.predict(full, val)}
        for s in [c[2:] for c in val.data.columns if c.startswith("r_")]:
            recs[SOURCES[s]] = R.source_only(val, s)
        for name, r in recs.items():
            bias.append({"задача": task, "модель": name, **organic_split(r, tg)})
            print(bias[-1], flush=True)
        base = per_user(recs["ранкер"], tg, 10)
        for s in [c[2:] for c in val.data.columns if c.startswith("r_")]:
            d2, v2 = R.without_source(dev, s), R.without_source(val, s)
            m = R.fit(d2)
            pu = per_user(R.predict(m, v2), tg, 10)
            d, lo, hi = paired_bootstrap(base, pu, "ndcg@10")
            abl.append({"задача": task, "без источника": SOURCES[s], "ndcg@10": round(float(pu["ndcg@10"].mean()), 4),
                        "Δ к полному": round(d, 4), "ДИ от": round(lo, 4), "ДИ до": round(hi, 4),
                        "полнота кандидатов": round(R.candidate_recall(v2), 4),
                        "кандидатов на польз.": round(v2.data.height / v2.data["uid"].n_unique())})
            print(abl[-1], flush=True)
    OUT.mkdir(exist_ok=True)
    pl.DataFrame(bias).write_csv(OUT / "organic_val.csv")
    pl.DataFrame(abl).write_csv(OUT / "ablation_val.csv")


if __name__ == "__main__":
    main()
