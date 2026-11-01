"""Загрузка данных Rosbank Churn и синтетика той же структуры (только для проверки кода).

Внутренний формат таблицы транзакций:
    client   id клиента
    days     сколько дней прошло от транзакции до момента прогноза (>= 0)
    amount   сумма
    mcc      код MCC (int)
    trx_category, channel_type   категориальные поля, если есть
И отдельная таблица меток: client, y.

Момент прогноза общий для всех клиентов: день после последней транзакции в датасете.
Брать для каждого клиента его собственную последнюю транзакцию нельзя: у ушедших клиентов
история обрывается раньше, и такой выбор отсчёта сам по себе выдаёт метку.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd


def _find_rosbank_csv(root: str) -> str:
    for path in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if path.endswith((".csv", ".csv.gz", ".parquet")):
            head = pd.read_parquet(path).head(1) if path.endswith(".parquet") else pd.read_csv(path, nrows=1)
            if {"cl_id", "target_flag"}.issubset(head.columns):
                return path
    raise FileNotFoundError(f"не нашёл файл с колонками cl_id и target_flag в {root}")


def load_rosbank(local_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rosbank Churn. Если local_dir не задан, скачивает датасет dllllb/rosbank-churn с Hugging Face."""
    if local_dir is None:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id="dllllb/rosbank-churn", repo_type="dataset")
    path = _find_rosbank_csv(local_dir)
    raw = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    return from_rosbank_frame(raw)


def from_rosbank_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(raw["TRDATETIME"], format="%d%b%y:%H:%M:%S", errors="coerce")
    if ts.isna().mean() > 0.5:  # на случай другого формата даты
        ts = pd.to_datetime(raw["TRDATETIME"], errors="coerce")
    raw = raw.assign(_ts=ts).dropna(subset=["_ts"])
    cutoff = raw["_ts"].max().normalize() + pd.Timedelta(days=1)
    trx = pd.DataFrame({
        "client": raw["cl_id"].to_numpy(),
        "days": (cutoff - raw["_ts"]).dt.total_seconds().to_numpy() / 86400.0,
        "amount": raw["amount"].astype(float).to_numpy(),
        "mcc": raw["MCC"].astype(int).to_numpy(),
    })
    for col in ("trx_category", "channel_type"):
        if col in raw.columns:
            trx[col] = raw[col].astype(str).to_numpy()
    labels = (raw.groupby("cl_id")["target_flag"].first().astype(int)
              .rename("y").rename_axis("client").reset_index())
    return trx, labels


def make_synthetic(n_clients: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Синтетика в той же схеме. Нужна только чтобы проверить, что код работает.
    Никаких выводов о пользе LLM по ней делать нельзя: сигнал в неё заложен вручную."""
    rng = np.random.default_rng(seed)
    mccs = np.array([5411, 5812, 5814, 6011, 4829, 5912, 5541, 4111, 4814, 5691, 5732, 5999, 7995, 4121, 5311])
    base_p = np.array([25, 8, 8, 6, 5, 4, 5, 6, 3, 4, 2, 6, 0.5, 3, 4], float)
    cats = np.array(["POS", "WD_ATM_ROS", "C2C_OUT", "DEPOSIT"])
    rows, labels = [], []
    for c in range(n_clients):
        churn = rng.random() < 0.35
        rate = rng.gamma(2.0, 0.4)                     # транзакций в день
        span = int(rng.integers(120, 400))             # длина истории в днях
        p = base_p * rng.dirichlet(np.ones(len(mccs)) * 2) * len(mccs)
        n = rng.poisson(rate * span) + 3
        days = rng.uniform(0, span, n)
        if churn and rng.random() < 0.4:  # часть уходящих клиентов постепенно перестаёт пользоваться картой
            keep = rng.random(n) < np.clip(days / rng.uniform(60, 200), 0.3, 1.0)
            days = days[keep] if keep.sum() > 2 else days[:3]
        if rng.random() < 0.1:            # шум в метках
            churn = not churn
        m = len(days)
        mcc = rng.choice(mccs, m, p=p / p.sum())
        cat_p = [0.75, 0.08, 0.12, 0.05] if churn else [0.80, 0.08, 0.07, 0.05]
        cat = rng.choice(cats, m, p=cat_p)
        amount = np.round(np.exp(rng.normal(6.5, 1.0, m)), 2)
        rows.append(pd.DataFrame({"client": c, "days": days, "amount": amount, "mcc": mcc,
                                  "trx_category": cat, "channel_type": "type1"}))
        labels.append((c, int(churn)))
    trx = pd.concat(rows, ignore_index=True)
    return trx, pd.DataFrame(labels, columns=["client", "y"])


def split_clients(labels: pd.DataFrame, seed: int = 0, fractions=(0.6, 0.2, 0.2)) -> dict[str, np.ndarray]:
    """Стратифицированное разбиение клиентов на search-train, search-val и test."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, grp in labels.groupby("y"):
        ids = grp["client"].to_numpy().copy()
        rng.shuffle(ids)
        a = int(len(ids) * fractions[0])
        b = a + int(len(ids) * fractions[1])
        parts["train"].append(ids[:a])
        parts["val"].append(ids[a:b])
        parts["test"].append(ids[b:])
    return {k: np.sort(np.concatenate(v)) for k, v in parts.items()}
