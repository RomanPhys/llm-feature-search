"""Грамматика признаков и её компиляция в векторизованный расчёт.

Признак задаётся словарём (JSON):
    agg      count | sum | mean | std | max | days_since_last | active_days | nunique_mcc
    value    amount | log_amount              (для sum, mean, std, max)
    window   7 | 30 | 90 | 180 | null         (null значит вся история)
    filter   null или {"mcc_group": имя} | {"mcc_in": [коды]} | {"trx_category": [значения]}
    norm     none | share_of_total | ratio_to_longer
    norm_window  длинное окно для ratio_to_longer

Окно всегда заканчивается в момент прогноза, поэтому заглянуть в будущее признак не может
по построению. Метки в грамматике не участвуют вообще.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

AGGS = ("count", "sum", "mean", "std", "max", "days_since_last", "active_days", "nunique_mcc")
VALUE_AGGS = ("sum", "mean", "std", "max")
VALUES = ("amount", "log_amount")
WINDOWS = (7, 30, 90, 180, None)
NORMS = ("none", "share_of_total", "ratio_to_longer")
SHARE_OK = ("count", "sum", "active_days")          # где доля от общего имеет смысл
RATIO_OK = ("count", "sum", "mean", "max", "active_days", "nunique_mcc")


class InvalidSpec(ValueError):
    pass


@dataclass
class Context:
    """Всё, что нужно для расчёта: транзакции, словарь групп MCC и допустимые значения полей."""
    trx: pd.DataFrame
    clients: np.ndarray
    mcc_groups: dict[str, list[int]]

    def __post_init__(self):
        self.trx = self.trx[self.trx["client"].isin(self.clients)].reset_index(drop=True)
        self.client_index = pd.Index(np.sort(self.clients))
        self.log_amount = np.log1p(np.abs(self.trx["amount"].to_numpy()))
        self.vocab = {c: set(self.trx[c].unique()) for c in ("trx_category", "channel_type") if c in self.trx}
        self.mcc_vocab = set(self.trx["mcc"].unique())
        self._cache: dict[str, np.ndarray] = {}


def spec_key(spec: dict) -> str:
    core = {k: spec.get(k) for k in ("agg", "value", "window", "filter", "norm", "norm_window")}
    return hashlib.md5(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:10]


def validate(spec: dict, ctx: Context) -> dict:
    """Проверяет спецификацию и возвращает нормализованную копию либо бросает InvalidSpec."""
    s = dict(spec)
    s.setdefault("value", "amount")
    s.setdefault("window", None)
    s.setdefault("filter", None)
    s.setdefault("norm", "none")
    if s["agg"] not in AGGS:
        raise InvalidSpec(f"неизвестный agg {s['agg']}")
    if s["agg"] in VALUE_AGGS and s["value"] not in VALUES:
        raise InvalidSpec(f"неизвестный value {s['value']}")
    if s["window"] not in WINDOWS:
        raise InvalidSpec(f"окно {s['window']} не из {WINDOWS}")
    if s["norm"] not in NORMS:
        raise InvalidSpec(f"неизвестная нормировка {s['norm']}")
    if s["norm"] == "share_of_total" and (s["agg"] not in SHARE_OK or not s["filter"]):
        raise InvalidSpec("share_of_total нужен фильтр и agg из " + str(SHARE_OK))
    if s["norm"] == "ratio_to_longer":
        lw = s.get("norm_window")
        if s["agg"] not in RATIO_OK or lw not in WINDOWS or s["window"] is None:
            raise InvalidSpec("ratio_to_longer: нужно короткое окно, длинное norm_window и agg из " + str(RATIO_OK))
        if lw is not None and lw <= s["window"]:
            raise InvalidSpec("norm_window должно быть длиннее window")
    f = s["filter"]
    if f:
        if len(f) != 1:
            raise InvalidSpec("в фильтре ровно одно условие")
        (kind, val), = f.items()
        if kind == "mcc_group":
            if val not in ctx.mcc_groups:
                raise InvalidSpec(f"нет группы MCC {val}")
            if not set(ctx.mcc_groups[val]) & ctx.mcc_vocab:
                raise InvalidSpec(f"коды группы {val} не встречаются в данных")
        elif kind == "mcc_in":
            if not set(map(int, val)) & ctx.mcc_vocab:
                raise InvalidSpec("коды mcc_in не встречаются в данных")
        elif kind in ("trx_category", "channel_type"):
            if kind not in ctx.vocab or not set(val) & ctx.vocab[kind]:
                raise InvalidSpec(f"значений {val} нет в поле {kind}")
        else:
            raise InvalidSpec(f"неизвестный фильтр {kind}")
    return s


def _mask(ctx: Context, window, flt) -> np.ndarray:
    m = np.ones(len(ctx.trx), bool) if window is None else (ctx.trx["days"].to_numpy() < window)
    if flt:
        (kind, val), = flt.items()
        if kind == "mcc_group":
            m &= ctx.trx["mcc"].isin(ctx.mcc_groups[val]).to_numpy()
        elif kind == "mcc_in":
            m &= ctx.trx["mcc"].isin([int(v) for v in val]).to_numpy()
        else:
            m &= ctx.trx[kind].isin(val).to_numpy()
    return m


def _raw(ctx: Context, agg, value, window, flt) -> np.ndarray:
    key = json.dumps([agg, value, window, flt], sort_keys=True, default=str)
    if key in ctx._cache:
        return ctx._cache[key]
    m = _mask(ctx, window, flt)
    sub = ctx.trx.loc[m]
    g = sub.groupby("client")
    if agg == "count":
        r = g.size()
    elif agg == "days_since_last":
        r = g["days"].min()
    elif agg == "active_days":
        r = np.floor(sub["days"]).groupby(sub["client"]).nunique()
    elif agg == "nunique_mcc":
        r = g["mcc"].nunique()
    else:
        vals = pd.Series(ctx.log_amount[m] if value == "log_amount" else sub["amount"].to_numpy(),
                         index=sub.index).groupby(sub["client"])
        r = getattr(vals, agg)()
    out = r.reindex(ctx.client_index).to_numpy(dtype=float)
    if agg in ("count", "sum", "active_days", "nunique_mcc"):
        out = np.nan_to_num(out, nan=0.0)        # нет событий значит ноль, а не пропуск
    ctx._cache[key] = out
    return out


def compute(spec: dict, ctx: Context) -> np.ndarray:
    """Значение признака для всех клиентов ctx.client_index (NaN, где не определено)."""
    s = validate(spec, ctx)
    x = _raw(ctx, s["agg"], s["value"], s["window"], s["filter"])
    if s["norm"] == "share_of_total":
        tot = _raw(ctx, s["agg"], s["value"], s["window"], None)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(tot > 0, x / tot, np.nan)
    elif s["norm"] == "ratio_to_longer":
        lng = _raw(ctx, s["agg"], s["value"], s["norm_window"], s["filter"])
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(np.abs(lng) > 0, x / lng, np.nan)
    return x


def describe(spec: dict) -> str:
    w = "вся история" if spec.get("window") is None else f"{spec['window']} дн."
    f = spec.get("filter")
    fs = "все транзакции" if not f else ", ".join(f"{k}={v}" for k, v in f.items())
    n = spec.get("norm", "none")
    tail = {"none": "", "share_of_total": ", доля от всех", "ratio_to_longer": f", к окну {spec.get('norm_window')} дн."}[n]
    val = f"({spec.get('value', 'amount')})" if spec["agg"] in VALUE_AGGS else ""
    return f"{spec['agg']}{val} | {w} | {fs}{tail}"


def base_features(ctx: Context | None = None, strong: bool = False, top_mcc: int = 10) -> list[dict]:
    """F0: набор агрегатов, с которого стартуют все методы одинаково.

    basic:  объём, суммы и активность без разбивки по категориям (15 признаков).
    strong: плюс доли операций по каждому значению категориальных полей и по самым частым MCC,
            как в стандартных агрегатах из литературы. Это честнее: признак, который просто
            повторяет известную разбивку, не должен считаться находкой."""
    out = []
    for w in (30, 90, None):
        out += [{"agg": "count", "window": w}, {"agg": "sum", "window": w},
                {"agg": "mean", "window": w}, {"agg": "std", "window": w}]
    out += [{"agg": "days_since_last"}, {"agg": "active_days", "window": 90},
            {"agg": "nunique_mcc", "window": None}]
    if strong:
        if ctx is None:
            raise ValueError("для strong нужен ctx")
        for field, vals in ctx.vocab.items():
            if len(vals) < 2:
                continue
            for v in sorted(vals):
                out += [{"agg": "count", "filter": {field: [v]}, "norm": "share_of_total"},
                        {"agg": "sum", "value": "amount", "filter": {field: [v]}, "norm": "share_of_total"}]
        for c in ctx.trx["mcc"].value_counts().index[:top_mcc]:
            out.append({"agg": "count", "filter": {"mcc_in": [int(c)]}, "norm": "share_of_total"})
    return out


def feature_matrix(specs: list[dict], ctx: Context) -> pd.DataFrame:
    cols = {}
    for s in specs:
        cols[s.get("name") or spec_key(s)] = compute(s, ctx)
    return pd.DataFrame(cols, index=ctx.client_index)
