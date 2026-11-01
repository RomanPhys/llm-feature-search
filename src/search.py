"""Источники кандидатов: случайный поиск в грамматике и LLM.

Чтобы отделить вклад LLM от вклада вычислений, у всех источников одинаковые
число кандидатов, грамматика, базовый набор F0 и правило отбора. Различается только то,
кто выбирает кандидатов.
"""
from __future__ import annotations

import json
import re

import numpy as np

from .dsl import AGGS, RATIO_OK, SHARE_OK, VALUE_AGGS, WINDOWS, Context, InvalidSpec, validate


def random_spec(rng: np.random.Generator, filters: list[dict]) -> dict:
    agg = str(rng.choice(AGGS))
    s = {"agg": agg, "window": WINDOWS[rng.integers(len(WINDOWS))]}
    if agg in VALUE_AGGS:
        s["value"] = str(rng.choice(["amount", "log_amount"]))
    if filters and rng.random() < 0.7:
        s["filter"] = filters[rng.integers(len(filters))]
    norms = ["none"]
    if s.get("filter") and agg in SHARE_OK:
        norms.append("share_of_total")
    if s["window"] is not None and agg in RATIO_OK:
        norms.append("ratio_to_longer")
    s["norm"] = str(rng.choice(norms))
    if s["norm"] == "ratio_to_longer":
        longer = [w for w in WINDOWS if w is None or w > s["window"]]
        s["norm_window"] = longer[rng.integers(len(longer))]
    return s


def random_pool(n: int, ctx: Context, filters: list[dict], seed: int) -> list[dict]:
    """n валидных уникальных случайных признаков из грамматики."""
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        s = random_spec(rng, filters)
        try:
            s = validate(s, ctx)
        except InvalidSpec:
            continue
        key = json.dumps(s, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def raw_filters(ctx: Context, top_mcc: int = 40) -> list[dict]:
    """Фильтры без знания смысла: отдельные частые коды MCC и значения категориальных полей."""
    top = ctx.trx["mcc"].value_counts().index[:top_mcc]
    return [{"mcc_in": [int(c)]} for c in top] + field_filters(ctx)


def field_filters(ctx: Context) -> list[dict]:
    """Фильтры по значениям категориальных полей кроме MCC (тип операции, канал)."""
    out = []
    for field, vals in ctx.vocab.items():
        if len(vals) >= 2:
            out += [{field: [v]} for v in sorted(vals)]
    return out


def group_filters(ctx: Context) -> list[dict]:
    """Смысловые группы MCC от LLM вместо отдельных кодов, остальные фильтры те же, что у raw_filters.
    Так два случайных пула отличаются только тем, как описаны коды MCC."""
    return [{"mcc_group": g} for g in ctx.mcc_groups] + field_filters(ctx)


def load_proposals(path: str) -> tuple[dict[str, list[int]], list[dict]]:
    """Файл с ответом LLM: группы MCC и список спецификаций признаков."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    groups = {g: expand_codes(codes) for g, codes in data["mcc_groups"].items()}
    return groups, data["features"]


def expand_codes(codes: list) -> list[int]:
    """Коды MCC могут быть числами или диапазонами вида "3000-3299"."""
    out = []
    for c in codes:
        if isinstance(c, str) and "-" in c:
            a, b = map(int, c.split("-"))
            out += list(range(a, b + 1))
        else:
            out.append(int(c))
    return out


def valid_specs(specs: list[dict], ctx: Context) -> tuple[list[dict], list[tuple[dict, str]]]:
    ok, bad = [], []
    for s in specs:
        try:
            ok.append(validate(s, ctx))
        except InvalidSpec as e:
            bad.append((s, str(e)))
    return ok, bad


# ---------------------------------------------------------------- LLM (необязательно)

def build_prompt(task: str, ctx: Context, stats: str, n: int, template_path: str) -> str:
    with open(template_path, encoding="utf-8") as f:
        tpl = f.read()
    return (tpl.replace("{TASK}", task).replace("{N}", str(n)).replace("{STATS}", stats)
               .replace("{FIELDS}", ", ".join(sorted(ctx.vocab)) or "нет"))


def ask_llm(prompt: str, model: str, base_url: str | None = None, api_key: str | None = None,
            temperature: float = 0.7) -> dict:
    """Любой OpenAI-совместимый endpoint (vLLM, Ollama, облачные API)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key or "none")
    resp = client.chat.completions.create(model=model, temperature=temperature,
                                          messages=[{"role": "user", "content": prompt}])
    text = resp.choices[0].message.content
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("в ответе LLM нет JSON")
    return json.loads(match.group(0))
