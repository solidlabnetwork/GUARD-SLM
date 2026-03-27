import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import joblib


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def sanitize_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def infer_model_name_from_input_path(input_path: str) -> str:
    base = os.path.basename(input_path)
    base_no_ext = os.path.splitext(base)[0]
    marker = "_activation_data_"
    if marker in base_no_ext:
        return base_no_ext.split(marker)[0]
    return base_no_ext


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _parse_csv_set(s):
    if s is None:
        return None
    items = [x.strip().lower() for x in s.split(",") if x.strip()]
    return set(items) if items else None


def remap_binary_type(original_type: str) -> int:
    """
    Binary label mapping:
      benign -> 0
      everything else -> 1
    """
    t = original_type.strip().lower()
    return 0 if t == "benign" else 1


# --------------------------------------------------
# Args
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Activation JSONL file")
    p.add_argument("--layer", type=int, required=True, help="Layer index to use")

    # Data controls
    p.add_argument("--max-samples-per-type", type=int, default=None,
                   help="Optional cap per original type. Default: use all.")
    p.add_argument("--include-types", type=str, default=None,
                   help="Comma-separated original types to include. Default: all. Case-insensitive.")
    p.add_argument("--exclude-types", type=str, default=None,
                   help="Comma-separated original types to exclude. Default: none. Case-insensitive.")

    # Model params
    p.add_argument("--C", type=float, default=1.0, help="SVM C")
    p.add_argument("--gamma", type=str, default="scale", help="SVM gamma (e.g., scale, auto, 0.1)")
    p.add_argument("--probability", action="store_true",
                   help="Enable predict_proba (slower, larger model).")

    # Saving
    p.add_argument("--outdir", type=str, default="./saved_models", help="Directory to save the model")
    p.add_argument("--model-name", type=str, default=None,
                   help="Base filename without extension. Default: auto from input path + layer.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files if present")

    return p.parse_args()


# --------------------------------------------------
# Load binary activations
# --------------------------------------------------
def load_binary_hidden_states(
    path: str,
    layer: int,
    include_types=None,
    exclude_types=None,
    max_samples_per_type=None,
):
    X, y = [], []

    counts_by_type = defaultdict(int)
    counts_by_binary_label = defaultdict(int)

    def allowed(t: str) -> bool:
        if include_types is not None and t not in include_types:
            return False
        if exclude_types is not None and t in exclude_types:
            return False
        return True

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading activations"):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if obj.get("layer", None) != layer:
                continue

            original_t = obj.get("type", None)
            if not isinstance(original_t, str) or not original_t.strip():
                continue
            original_t = original_t.strip().lower()

            if not allowed(original_t):
                continue

            if max_samples_per_type is not None and counts_by_type[original_t] >= max_samples_per_type:
                continue

            act = obj.get("activation", None)
            if not isinstance(act, list) or len(act) == 0:
                continue

            label = remap_binary_type(original_t)

            X.append(act)
            y.append(label)

            counts_by_type[original_t] += 1
            counts_by_binary_label[label] += 1

    if len(X) == 0:
        raise RuntimeError("No samples loaded. Check --layer, input path, or filters.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if X.ndim != 2:
        raise RuntimeError(f"Expected 2D feature matrix, got shape {X.shape}.")

    feat_dim = X.shape[1]

    print("\n[Loaded]")
    print(f"Layer: {layer}")
    print(f"Total samples: {len(X)} | feat_dim: {feat_dim}")
    print("Binary label counts:", {
        "benign(0)": int((y == 0).sum()),
        "nonbenign(1)": int((y == 1).sum()),
    })

    if "jbkv" in counts_by_type:
        print(f"JBKV count (mapped to nonbenign=1): {counts_by_type['jbkv']}")

    top_types = sorted(counts_by_type.items(), key=lambda kv: kv[1], reverse=True)[:20]
    print("\nTop original types (up to 20):")
    for tt, c in top_types:
        print(f"  {tt:>14}: {c}")

    return X, y, counts_by_type, counts_by_binary_label, feat_dim


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()

    include_types = _parse_csv_set(args.include_types)
    exclude_types = _parse_csv_set(args.exclude_types)

    ensure_dir(args.outdir)

    inferred_model = infer_model_name_from_input_path(args.input)
    inferred_model = sanitize_name(inferred_model)

    model_name = args.model_name
    if model_name is None:
        model_name = f"{inferred_model}_svm_rbf_binary_layer{args.layer}"
    model_name = sanitize_name(model_name)

    model_path = os.path.join(args.outdir, model_name + ".joblib")
    meta_path = os.path.join(args.outdir, model_name + ".meta.json")

    if not args.overwrite:
        for pth in [model_path, meta_path]:
            if os.path.exists(pth):
                raise FileExistsError(
                    f"File already exists: {pth}\n"
                    f"Use --overwrite or choose a different --model-name."
                )

    X, y, counts_by_type, counts_by_binary_label, feat_dim = load_binary_hidden_states(
        args.input,
        args.layer,
        include_types=include_types,
        exclude_types=exclude_types,
        max_samples_per_type=args.max_samples_per_type,
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            gamma=args.gamma,
            C=args.C,
            probability=bool(args.probability),
            decision_function_shape="ovr",
        )),
    ])

    print("\nTraining binary RBF SVM on FULL data...")
    clf.fit(X, y)

    joblib.dump(clf, model_path)

    meta = {
        "input": args.input,
        "layer": args.layer,
        "feature_dim": int(feat_dim),
        "model_name": model_name,
        "label_mapping": {
            "benign": 0,
            "non_benign": 1
        },
        "counts_by_type": dict(counts_by_type),
        "counts_by_binary_label": {
            "benign(0)": int(counts_by_binary_label.get(0, 0)),
            "nonbenign(1)": int(counts_by_binary_label.get(1, 0)),
        },
        "filters": {
            "include_types": sorted(list(include_types)) if include_types is not None else None,
            "exclude_types": sorted(list(exclude_types)) if exclude_types is not None else None,
            "max_samples_per_type": args.max_samples_per_type,
        },
        "svm_params": {
            "kernel": "rbf",
            "C": args.C,
            "gamma": args.gamma,
            "probability": bool(args.probability),
        },
        "pipeline": ["StandardScaler", "SVC(RBF)"],
        "notes": (
            "Binary mapping: only 'benign' is label 0. "
            "All other original types, including malicious, jbkv, autodan, gcg, "
            "pair, cipher, ica, tap, etc., are mapped to label 1."
        )
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[SAVED]")
    print("Model:", model_path)
    print("Meta :", meta_path)
    print("\nDone.")


if __name__ == "__main__":
    main()