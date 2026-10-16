#!/usr/bin/env python3
"""Single-file HE-XFL research scaffold.

Implements: CSV/synthetic data loading, train-only preprocessing, Dirichlet non-IID
partitioning, 50 edge clients / 5 fog nodes, local PyTorch training, hierarchical
sample-weighted FedAvg, held-out evaluation, and JSON/CSV logging.

Optional CKKS demo: --he-demo encrypts weighted client model vectors and performs
homomorphic addition in one process. This demonstrates arithmetic only; it is NOT
an end-to-end secure deployment or a complete key-management/threat-model solution.

Differential Privacy is NOT implemented in this script. Do not claim DP guarantees
unless DP-SGD and a validated accountant are separately integrated and tested.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------- Configuration -------------------------------
DEFAULTS = {
    "seed": 42,
    "num_clients": 50,
    "num_fog_nodes": 5,
    "dirichlet_alpha": 0.5,
    "rounds": 5,
    "local_epochs": 1,
    "batch_size": 64,
    "learning_rate": 0.001,
    "hidden_dims": [128, 64],
    "test_size": 0.15,
    "validation_size": 0.15,
    "synthetic_samples": 6000,
    "synthetic_features": 20,
    "synthetic_classes": 4,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------- Data ---------------------------------------
def load_raw_data(args):
    """Return X, y. CSV must contain a categorical/numeric target column."""
    if args.source == "synthetic":
        X, y = make_classification(
            n_samples=args.synthetic_samples,
            n_features=args.synthetic_features,
            n_informative=max(2, args.synthetic_features // 2),
            n_redundant=max(0, args.synthetic_features // 10),
            n_classes=args.synthetic_classes,
            n_clusters_per_class=1,
            random_state=args.seed,
        )
        return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])]), pd.Series(y, name="label")

    if not args.csv_path or not args.label_column:
        raise ValueError("For --source csv, provide both --csv-path and --label-column")
    df = pd.read_csv(args.csv_path)
    if args.label_column not in df.columns:
        raise ValueError(f"Label column {args.label_column!r} not found in CSV")
    y = df[args.label_column]
    X = df.drop(columns=[args.label_column])
    # This baseline accepts numeric features only; encode categorical predictors upstream.
    X = X.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError("No numeric predictor columns found in CSV")
    valid = y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def prepare_data(args):
    X_df, y_series = load_raw_data(args)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_series.astype(str))

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    val_fraction_of_remaining = args.validation_size / (1.0 - args.test_size)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_fraction_of_remaining,
        random_state=args.seed, stratify=y[train_idx]
    )

    # Fit imputer/scaler on training data only to prevent test/validation leakage.
    X_all = X_df.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_all[train_idx])).astype(np.float32)
    X_val = scaler.transform(imputer.transform(X_all[val_idx])).astype(np.float32)
    X_test = scaler.transform(imputer.transform(X_all[test_idx])).astype(np.float32)

    def as_dataset(X, idx):
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y[idx].astype(np.int64)))

    return (X_train, y[train_idx], as_dataset(X_val, val_idx), as_dataset(X_test, test_idx),
            len(label_encoder.classes_), label_encoder.classes_.tolist(), X_df.columns.tolist())


def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float, seed: int) -> List[np.ndarray]:
    """Class-wise Dirichlet partition; each training sample is assigned once."""
    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
        for client_id, part in enumerate(np.split(cls_idx, cuts)):
            client_indices[client_id].extend(part.tolist())
    result = [np.asarray(v, dtype=np.int64) for v in client_indices]
    # Empty partitions can occur with small datasets/extreme alpha; fail clearly.
    empty = [i for i, v in enumerate(result) if len(v) == 0]
    if empty:
        raise ValueError(f"Dirichlet partition produced empty clients {empty}; increase data or alpha")
    return result


# -------------------------------- Model ---------------------------------------
class IntrusionMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(prev, int(width)), nn.ReLU()])
            prev = int(width)
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def state_to_vector(state: Dict[str, torch.Tensor]) -> np.ndarray:
    return np.concatenate([v.detach().cpu().numpy().reshape(-1).astype(np.float64)
                           for v in state.values()])


def vector_to_state(vector: np.ndarray, template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result, offset = {}, 0
    for key, tensor in template.items():
        count = tensor.numel()
        arr = vector[offset:offset + count].reshape(tuple(tensor.shape))
        result[key] = torch.as_tensor(arr, dtype=tensor.dtype)
        offset += count
    if offset != len(vector):
        raise ValueError("Vector length does not match model state")
    return result


# ----------------------------- Local training ---------------------------------
def train_client(global_state, X, y, indices, input_dim, num_classes, args, device):
    model = IntrusionMLP(input_dim, num_classes, args.hidden_dims).to(device)
    model.load_state_dict(global_state)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    subset = TensorDataset(torch.from_numpy(X[indices]), torch.from_numpy(y[indices].astype(np.int64)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
    model.train()
    for _ in range(args.local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, len(indices)


def fedavg(states: List[Dict[str, torch.Tensor]], counts: List[int]) -> Dict[str, torch.Tensor]:
    if not states or len(states) != len(counts):
        raise ValueError("Provide matching non-empty states and sample counts")
    if any(int(n) <= 0 for n in counts):
        raise ValueError("Sample counts must be positive")
    total = float(sum(counts))
    out = {}
    for key in states[0]:
        out[key] = sum(sd[key].detach().cpu() * (float(n) / total)
                       for sd, n in zip(states, counts))
    return out


# ---------------------- Optional CKKS aggregation demo ------------------------
def ckks_aggregate_weighted_states(states, counts, template, poly_modulus_degree=8192):
    """Homomorphically add weighted flattened model states using TenSEAL CKKS.

    This single-process demo keeps the secret key in this process. It is not a
    complete secure multi-party deployment and should not be described as one.
    """
    try:
        import tenseal as ts
    except ImportError as exc:
        raise RuntimeError("Install TenSEAL first: pip install tenseal") from exc

    context = ts.context(ts.SCHEME_TYPE.CKKS,
                         poly_modulus_degree=poly_modulus_degree,
                         coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2 ** 40
    slot_count = poly_modulus_degree // 2
    total_count = float(sum(counts))
    encrypted_sum = None
    for state, count in zip(states, counts):
        vec = state_to_vector(state) * (float(count) / total_count)
        encrypted_chunks = []
        for start in range(0, len(vec), slot_count):
            chunk = vec[start:start + slot_count].tolist()
            encrypted_chunks.append(ts.ckks_vector(context, chunk))
        if encrypted_sum is None:
            encrypted_sum = encrypted_chunks
        else:
            encrypted_sum = [a + b for a, b in zip(encrypted_sum, encrypted_chunks)]
    decrypted = np.concatenate([np.asarray(chunk.decrypt(), dtype=np.float64)
                                for chunk in encrypted_sum])[:len(state_to_vector(template))]
    return vector_to_state(decrypted, template)


# -------------------------------- Evaluation ----------------------------------
def evaluate(model, dataset, device, num_classes, class_names):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.numpy().tolist())
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels,
            target_names=[str(c) for c in class_names], output_dict=True, zero_division=0),
        "test_samples": len(y_true),
    }


# ---------------------------------- Runner ------------------------------------
def main():
    p = argparse.ArgumentParser(description="Single-file HE-XFL hierarchical FL scaffold")
    p.add_argument("--source", choices=["synthetic", "csv"], default="synthetic")
    p.add_argument("--csv-path", default="")
    p.add_argument("--label-column", default="label")
    p.add_argument("--output-dir", default="he_xfl_results")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--clients", type=int, default=DEFAULTS["num_clients"])
    p.add_argument("--fog-nodes", type=int, default=DEFAULTS["num_fog_nodes"])
    p.add_argument("--alpha", type=float, default=DEFAULTS["dirichlet_alpha"])
    p.add_argument("--rounds", type=int, default=DEFAULTS["rounds"])
    p.add_argument("--local-epochs", type=int, default=DEFAULTS["local_epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--hidden-dims", type=int, nargs="+", default=DEFAULTS["hidden_dims"])
    p.add_argument("--synthetic-samples", type=int, default=DEFAULTS["synthetic_samples"])
    p.add_argument("--synthetic-features", type=int, default=DEFAULTS["synthetic_features"])
    p.add_argument("--synthetic-classes", type=int, default=DEFAULTS["synthetic_classes"])
    p.add_argument("--test-size", type=float, default=DEFAULTS["test_size"])
    p.add_argument("--validation-size", type=float, default=DEFAULTS["validation_size"])
    p.add_argument("--he-demo", action="store_true", help="Use single-process TenSEAL CKKS for aggregation")
    args = p.parse_args()

    if args.clients % args.fog_nodes != 0:
        raise ValueError("For this implementation, --clients must be divisible by --fog-nodes")
    if args.alpha <= 0 or args.rounds < 1 or args.local_epochs < 1:
        raise ValueError("alpha, rounds, and local epochs must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, val_ds, test_ds, num_classes, class_names, feature_names = prepare_data(args)
    partitions = dirichlet_partition(y_train, args.clients, args.alpha, args.seed)
    clients_per_fog = args.clients // args.fog_nodes
    fog_assignments = [list(range(f * clients_per_fog, (f + 1) * clients_per_fog))
                       for f in range(args.fog_nodes)]

    model = IntrusionMLP(X_train.shape[1], num_classes, args.hidden_dims).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    metadata = {
        "source": args.source,
        "csv_path": args.csv_path if args.source == "csv" else None,
        "label_column": args.label_column if args.source == "csv" else "label",
        "seed": args.seed, "num_clients": args.clients, "num_fog_nodes": args.fog_nodes,
        "clients_per_fog": clients_per_fog, "dirichlet_alpha": args.alpha,
        "rounds": args.rounds, "local_epochs": args.local_epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "hidden_dims": args.hidden_dims, "input_features": X_train.shape[1],
        "num_classes": num_classes, "class_names": [str(x) for x in class_names],
        "feature_names": feature_names, "device": str(device),
        "differential_privacy": "NOT IMPLEMENTED in this script",
        "homomorphic_encryption": "optional single-process CKKS demo only" if args.he_demo else "disabled",
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    print(f"Device={device}; clients={args.clients}; fog nodes={args.fog_nodes}; clients/fog={clients_per_fog}")
    print(f"Training data={len(y_train)}, validation={len(val_ds)}, test={len(test_ds)}")

    for rnd in range(1, args.rounds + 1):
        start_time = time.time()
        fog_states, fog_counts = [], []
        for fog_id, client_ids in enumerate(fog_assignments):
            local_states, local_counts = [], []
            for client_id in client_ids:
                state, count = train_client(global_state, X_train, y_train,
                    partitions[client_id], X_train.shape[1], num_classes, args, device)
                local_states.append(state)
                local_counts.append(count)
            if args.he_demo:
                fog_state = ckks_aggregate_weighted_states(local_states, local_counts, global_state)
            else:
                fog_state = fedavg(local_states, local_counts)
            fog_states.append(fog_state)
            fog_counts.append(sum(local_counts))
            print(f"Round {rnd}/{args.rounds} | Fog {fog_id + 1}/{args.fog_nodes} aggregated {len(client_ids)} clients")

        if args.he_demo:
            global_state = ckks_aggregate_weighted_states(fog_states, fog_counts, global_state)
        else:
            global_state = fedavg(fog_states, fog_counts)

        model.load_state_dict(global_state)
        val_metrics = evaluate(model, val_ds, device, num_classes, class_names)
        test_metrics = evaluate(model, test_ds, device, num_classes, class_names)
        record = {
            "round": rnd,
            "duration_seconds": round(time.time() - start_time, 3),
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "test_samples": test_metrics["test_samples"],
        }
        history.append(record)
        print(f"Round {rnd}: test accuracy={record['test_accuracy']:.4f}, "
              f"macro-F1={record['test_macro_f1']:.4f}, "
              f"weighted-F1={record['test_weighted_f1']:.4f}")
        torch.save(global_state, output_dir / f"global_model_round_{rnd}.pt")
        (output_dir / f"evaluation_round_{rnd}.json").write_text(json.dumps(test_metrics, indent=2))

    with (output_dir / "round_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "final_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (output_dir / "class_names.json").write_text(json.dumps([str(x) for x in class_names], indent=2))
    print(f"\nCompleted. Results saved to: {output_dir.resolve()}")
    print("IMPORTANT: synthetic results are smoke-test results, not Edge-IIoTset/Kitsune results.")
    if args.he_demo:
        print("IMPORTANT: CKKS mode is a single-process arithmetic demo; key custody and distributed security are not implemented.")
    print("IMPORTANT: Differential Privacy is not implemented; do not claim an (epsilon, delta) guarantee from this script.")


if __name__ == "__main__":
    main()
