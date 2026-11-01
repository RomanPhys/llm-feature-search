"""Проверки ключевых свойств: причинность окон, нормировки, процедура Холма.
Запуск: python tests/test_core.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dsl import Context, InvalidSpec, compute, validate  # noqa: E402
from src.evaluate import holm  # noqa: E402
from src.search import expand_codes, random_pool, raw_filters  # noqa: E402


def toy():
    trx = pd.DataFrame({
        "client": [1, 1, 1, 2, 2],
        "days":   [1, 10, 100, 5, 40],
        "amount": [100, 200, 300, 50, 70],
        "mcc":    [5411, 6011, 5411, 5411, 4829],
        "trx_category": ["POS", "WD_ATM_ROS", "POS", "POS", "C2C_OUT"],
    })
    return Context(trx, np.array([1, 2, 3]), {"groceries": [5411], "cash": [6011]})


def test_window_is_causal():
    ctx = toy()
    x = compute({"agg": "count", "window": 30}, ctx)
    assert list(x) == [2, 1, 0]                  # события 100 и 40 дней назад в окно 30 дней не входят


def test_share_and_ratio():
    ctx = toy()
    share = compute({"agg": "sum", "value": "amount", "window": None,
                     "filter": {"mcc_group": "groceries"}, "norm": "share_of_total"}, ctx)
    assert np.isclose(share[0], 400 / 600) and np.isclose(share[1], 50 / 120) and np.isnan(share[2])
    ratio = compute({"agg": "count", "window": 30, "norm": "ratio_to_longer", "norm_window": None}, ctx)
    assert np.isclose(ratio[0], 2 / 3) and np.isclose(ratio[1], 1 / 2)


def test_days_since_last_and_filters():
    ctx = toy()
    d = compute({"agg": "days_since_last", "filter": {"trx_category": ["C2C_OUT"]}}, ctx)
    assert np.isnan(d[0]) and d[1] == 40 and np.isnan(d[2])


def test_invalid_specs_rejected():
    ctx = toy()
    for bad in [{"agg": "median"}, {"agg": "count", "window": 45},
                {"agg": "count", "window": 30, "norm": "ratio_to_longer", "norm_window": 7},
                {"agg": "count", "filter": {"mcc_group": "nope"}},
                {"agg": "count", "filter": {"trx_category": ["NOPE"]}}]:
        try:
            validate(bad, ctx)
        except InvalidSpec:
            continue
        raise AssertionError(f"спецификация должна быть отклонена: {bad}")


def test_holm():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert holm(p, 0.05).tolist() == [True, False, False, True]   # 0.005<=0.0125, 0.01<=0.0167, 0.03>0.025


def test_random_pool_valid_and_unique():
    ctx = toy()
    pool = random_pool(15, ctx, raw_filters(ctx), seed=0)
    assert len(pool) == 15 and len({str(sorted(s.items())) for s in pool}) == 15


def test_expand_codes():
    assert expand_codes([5411, "3000-3002"]) == [5411, 3000, 3001, 3002]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
