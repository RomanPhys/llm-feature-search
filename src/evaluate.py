"""Оценка кандидатов: прирост ROC-AUC, парный бутстрэп, поправка Холма и финальная проверка на тесте."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def make_model(seed: int = 0):
    """LightGBM, если установлен (в Colab он есть), иначе HistGradientBoosting из sklearn."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=seed, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              min_samples_leaf=20, random_state=seed)


def fit_predict(X_tr, y_tr, X_ev, seed: int = 0) -> np.ndarray:
    m = make_model(seed)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_ev)[:, 1]


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    return roc_auc_score(y, p)


def paired_bootstrap(y, p_new, p_base, n_boot: int = 500, seed: int = 0) -> dict:
    """Разница AUC (новая минус базовая) на одних и тех же пересэмплированных клиентах."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            deltas[b] = 0.0
            continue
        deltas[b] = fast_auc(yb, p_new[idx]) - fast_auc(yb, p_base[idx])
    return {
        "delta": fast_auc(y, p_new) - fast_auc(y, p_base),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "p_value": float((deltas <= 0).mean()),          # односторонний: прироста нет
    }


def holm(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Маска отвергнутых гипотез прироста нет по процедуре Холма."""
    p = np.asarray(p_values)
    order = np.argsort(p)
    m = len(p)
    reject = np.zeros(m, bool)
    for k, i in enumerate(order):
        if p[i] <= alpha / (m - k):
            reject[i] = True
        else:
            break
    return reject


def score_pool(specs: list[dict], F_cand: pd.DataFrame, X0: pd.DataFrame, y: pd.Series,
               ids: dict, n_boot: int = 500, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Для каждого кандидата: модель на F0 + кандидат против модели на F0, оценка на search-val."""
    tr, va = ids["train"], ids["val"]
    p_base = fit_predict(X0.loc[tr], y.loc[tr], X0.loc[va], seed)
    yv = y.loc[va].to_numpy()
    rows = []
    for col, spec in zip(F_cand.columns, specs):
        Xtr = X0.loc[tr].assign(_c=F_cand.loc[tr, col])
        Xva = X0.loc[va].assign(_c=F_cand.loc[va, col])
        p_new = fit_predict(Xtr, y.loc[tr], Xva, seed)
        r = paired_bootstrap(yv, p_new, p_base, n_boot, seed)
        rows.append({"feature": col, **r, "spec": spec})
    return pd.DataFrame(rows), p_base


def select(table: pd.DataFrame, F_cand: pd.DataFrame, rule: str, k_max: int = 10,
           alpha: float = 0.05, corr_max: float = 0.95) -> list[str]:
    """Правило отбора.
    naive:   принять, если средний прирост на валидации больше нуля (как в CAAFE);
    strict:  принять, если прирост значим по бутстрэпу после поправки Холма."""
    if rule == "naive":
        ok = table["delta"].to_numpy() > 0
    elif rule == "strict":
        ok = holm(table["p_value"].to_numpy(), alpha)
    else:
        raise ValueError(rule)
    cand = table[ok].sort_values("delta", ascending=False)["feature"].tolist()
    chosen = []
    for c in cand:
        x = F_cand[c]
        if x.nunique(dropna=True) < 2:          # константный признак ничего не добавляет
            continue
        if all(abs(np.nan_to_num(x.corr(F_cand[d], method="spearman"))) < corr_max for d in chosen):
            chosen.append(c)
        if len(chosen) >= k_max:
            break
    return chosen


def val_gain(chosen: list[str], F_cand: pd.DataFrame, X0: pd.DataFrame, y: pd.Series, ids: dict,
             seed: int = 0) -> float:
    """Прирост AUC всего отобранного набора на search-val, то есть там, где его и отбирали."""
    if not chosen:
        return 0.0
    tr, va = ids["train"], ids["val"]
    yv = y.loc[va].to_numpy()
    p_base = fit_predict(X0.loc[tr], y.loc[tr], X0.loc[va], seed)
    X = X0.join(F_cand[chosen])
    p_new = fit_predict(X.loc[tr], y.loc[tr], X.loc[va], seed)
    return fast_auc(yv, p_new) - fast_auc(yv, p_base)


def test_gain(chosen: list[str], F_cand: pd.DataFrame, X0: pd.DataFrame, y: pd.Series, ids: dict,
              n_boot: int = 1000, seed: int = 0) -> dict:
    """Финал: обучение на train+val, одна проверка на test, парный бутстрэп против F0."""
    fit_ids = np.concatenate([ids["train"], ids["val"]])
    te = ids["test"]
    p_base = fit_predict(X0.loc[fit_ids], y.loc[fit_ids], X0.loc[te], seed)
    if not chosen:
        auc = fast_auc(y.loc[te].to_numpy(), p_base)
        return {"n_features": 0, "test_auc": auc, "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    X = X0.join(F_cand[chosen])
    p_new = fit_predict(X.loc[fit_ids], y.loc[fit_ids], X.loc[te], seed)
    yt = y.loc[te].to_numpy()
    r = paired_bootstrap(yt, p_new, p_base, n_boot, seed)
    return {"n_features": len(chosen), "test_auc": fast_auc(yt, p_new), **{k: r[k] for k in ("delta", "ci_low", "ci_high")}}


def run_pool(name: str, specs: list[dict], ctx, X0, y, ids, n_boot=500, seed=0, k_max=10) -> dict:
    """Полный прогон одного источника кандидатов с обоими правилами отбора."""
    from .dsl import feature_matrix, spec_key
    t0 = time.time()
    specs = [dict(s, name=f"{name}_{i}_{spec_key(s)}") for i, s in enumerate(specs)]
    F = feature_matrix(specs, ctx)
    table, _ = score_pool(specs, F, X0, y, ids, n_boot, seed)
    out = {"pool": name, "n_candidates": len(specs), "table": table}
    for rule in ("naive", "strict"):
        chosen = select(table, F, rule, k_max)
        tg = test_gain(chosen, F, X0, y, ids, seed=seed)
        out[rule] = {"chosen": chosen, "val_delta": val_gain(chosen, F, X0, y, ids, seed), **tg}
    out["seconds"] = time.time() - t0
    return out
