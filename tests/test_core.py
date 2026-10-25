"""Метрики, окна и отсутствие утечки будущего — на маленьких синтетических данных с известным ответом."""
import math

import polars as pl
import pytest

from recsys import baselines as B
from recsys import ranker as R
from recsys.data import history, seen_items, targets
from recsys.metrics import paired_bootstrap, per_user


def ev(rows):
    """rows: (uid, item, day, pos[, organic]) → таблица событий в формате load_events()."""
    out = []
    for r in rows:
        uid, item, day, pos = r[:4]
        org = r[4] if len(r) > 4 else True
        out.append({"uid": uid, "item_id": item, "ts": day * 17280, "day": day, "organic": org, "listen": True,
                    "pos": pos, "skip": not pos, "like": False, "dislike": False})
    return pl.DataFrame(out, schema_overrides={"uid": pl.UInt32, "item_id": pl.UInt32, "ts": pl.UInt32,
                                               "day": pl.UInt16})


def recs(rows):
    return pl.DataFrame(rows, schema=["uid", "item_id", "rank"], orient="row",
                        schema_overrides={"uid": pl.UInt32, "item_id": pl.UInt32, "rank": pl.UInt32})


def test_ndcg_and_recall_known_values():
    tg = pl.DataFrame({"uid": [1, 1, 2], "item_id": [10, 20, 30]}, schema_overrides={"uid": pl.UInt32, "item_id": pl.UInt32})
    r = recs([(1, 10, 1), (1, 99, 2), (1, 20, 3), (2, 77, 1)])
    u = per_user(r, tg, 10).sort("uid")
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert u["ndcg@10"][0] == pytest.approx(dcg / idcg)
    assert u["recall@10"][0] == 1.0
    assert u["ndcg@10"][1] == 0.0 and u["recall@10"][1] == 0.0       # пользователь без попаданий — ноль, а не пропуск


def test_user_without_recs_counts_as_zero():
    tg = pl.DataFrame({"uid": [1, 2], "item_id": [10, 20]}, schema_overrides={"uid": pl.UInt32, "item_id": pl.UInt32})
    u = per_user(recs([(1, 10, 1)]), tg, 10)
    assert u.height == 2 and u["ndcg@10"].mean() == pytest.approx(0.5)


def test_targets_window_and_discovery():
    e = ev([(1, 10, 5, True), (1, 10, 12, True), (1, 11, 12, True), (1, 12, 12, False), (1, 13, 20, True)])
    all_ = targets(e, 10, days=7)
    assert set(all_["item_id"]) == {10, 11}                          # 12 — не положительный, 13 — за окном
    new = targets(e, 10, new_only=True, days=7)
    assert set(new["item_id"]) == {11}                               # 10 уже встречался до окна
    assert set(seen_items(e, 10)["item_id"]) == {10}


def test_history_has_no_future():
    e = ev([(1, 10, 5, True), (1, 11, 15, True)])
    assert history(e, 10)["day"].max() < 10


def test_popular_excludes_seen_and_own_top_prefers_recent():
    e = ev([(1, 10, 1, True), (2, 10, 9, True), (3, 10, 9, True), (2, 11, 9, True), (1, 12, 9, True)])
    users = pl.Series("uid", [1], dtype=pl.UInt32)
    p = B.popular(e, users, 2, None, exclude=seen_items(e, 10))
    assert 10 not in set(p["item_id"]) and 12 not in set(p["item_id"])
    own = B.own_top(ev([(1, 10, 1, True), (1, 10, 2, True), (1, 11, 99, True)]), users, 2, half_life=5)
    assert own.sort("rank")["item_id"][0] == 11                       # свежий трек важнее старых повторов


def test_features_use_only_history():
    e = ev([(1, 10, 1, True), (2, 10, 2, True), (3, 10, 50, True)])   # третий отклик — в будущем
    hist = history(e, 10)
    cand = pl.DataFrame({"uid": [1], "item_id": [10]}, schema_overrides={"uid": pl.UInt32, "item_id": pl.UInt32})
    artists = pl.DataFrame({"artist_id": [1], "item_id": [10]}, schema_overrides={"artist_id": pl.UInt32,
                                                                                    "item_id": pl.UInt32})
    f = R.features(cand, hist, artists)
    assert f["i_pos_all"][0] == 2


def test_without_source_drops_exclusive_candidates():
    d = pl.DataFrame({"uid": [1, 1, 1], "item_id": [1, 2, 3], "r_a": [1.0, None, 2.0], "r_b": [None, 1.0, 1.0],
                      "label": [1, 0, 0]})
    w = R.without_source(R.Window(0, False, d, 1), "a")
    assert set(w.data["item_id"]) == {2, 3} and "r_a" not in w.data.columns


def test_paired_bootstrap_detects_shift():
    a = pl.DataFrame({"uid": range(500), "m": [0.1] * 500})
    b = a.with_columns(pl.col("m") + 0.05)
    d, lo, hi = paired_bootstrap(a, b, "m")
    assert d == pytest.approx(0.05) and lo == pytest.approx(0.05) and hi == pytest.approx(0.05)
