import argparse
import json
import os
import re
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


INDIVIDUAL_JB_CATEGORIES = {
    "autodan",
    "cipher",
    "codechamelon",
    "deepinception",
    "gcg",
    "ica",
    "jailbroken",
    "pair",
    "tap",
}


CATEGORY_COLORS = {
    "benign":        "#1f77b4",
    "malicious":     "#d62728",
    "autodan":       "#9467bd",
    "cipher":        "#17becf",
    "codechamelon":  "#8c564b",
    "deepinception": "#ff7f0e",
    "gcg":           "#bcbd22",
    "ica":           "#2ca02c",
    "jailbroken":    "#e377c2",
    "pair":          "#7f7f7f",
    "tap":           "#ff1493",
}


PREFERRED_ORDER = [
    "benign",
    "malicious",
    "autodan",
    "cipher",
    "codechamelon",
    "deepinception",
    "gcg",
    "ica",
    "jailbroken",
    "pair",
    "tap",
]


def sanitize_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def infer_model_name_from_input_path(input_path: str) -> str:
    base = os.path.basename(input_path)
    base_no_ext = os.path.splitext(base)[0]

    marker = "_activation_data_"
    if marker in base_no_ext:
        return base_no_ext.split(marker)[0]

    m = re.match(r"(.+)_\d{8}_\d{6}$", base_no_ext)
    if m:
        return m.group(1)

    return base_no_ext


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def stable_order_types(types_seen):
    seen = sorted(set(types_seen))
    ordered = [t for t in PREFERRED_ORDER if t in seen]
    tail = [t for t in seen if t not in set(PREFERRED_ORDER)]
    return ordered + tail


def normalize_type(t: str) -> str:
    return str(t).strip().lower()


def remap_type(original_type: str) -> str:
    t = normalize_type(original_type)

    if t == "benign":
        return "benign"

    if t in INDIVIDUAL_JB_CATEGORIES:
        return t

    return "malicious"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input", type=str, required=True,
                   help="Activation JSONL file")
    p.add_argument("--layer", type=int, required=True,
                   help="Layer index to analyze (0-based)")
    p.add_argument("--max-samples", type=int, default=2000,
                   help="Max samples per final class")

    p.add_argument("--outdir", type=str, default="figures",
                   help="Base directory for saving figures")
    p.add_argument("--model", type=str, default=None,
                   help="Optional model name")
    p.add_argument("--no-timestamp", action="store_true",
                   help="Do not add timestamp to output filename")

    p.add_argument("--include-types", type=str, default=None,
                   help="Comma-separated original types to include before remapping")
    p.add_argument("--exclude-types", type=str, default=None,
                   help="Comma-separated original types to exclude before remapping")

    p.add_argument("--perplexity", type=float, default=30.0,
                   help="tSNE perplexity")
    p.add_argument("--learning-rate", type=str, default="auto",
                   help="tSNE learning rate: 'auto' or a float")
    p.add_argument("--n-iter", type=int, default=1000,
                   help="tSNE iterations")
    p.add_argument("--init", type=str, default="pca", choices=["pca", "random"],
                   help="Initialization for tSNE")
    p.add_argument("--metric", type=str, default="euclidean",
                   help="Distance metric")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed")

    p.add_argument("--pca-dim", type=int, default=128,
                   help="PCA dimension before tSNE (0 disables)")
    p.add_argument("--center", action="store_true",
                   help="Center features before PCA/tSNE")
    p.add_argument("--equal-aspect", action="store_true",
                   help="Use equal axis aspect ratio")

    p.add_argument("--point-size", type=float, default=10.0,
                   help="Scatter point size")
    p.add_argument("--alpha", type=float, default=0.7,
                   help="Scatter alpha")

    p.add_argument("--xlabel-fontsize", type=int, default=30,
                   help="X label font size")
    p.add_argument("--ylabel-fontsize", type=int, default=30,
                   help="Y label font size")
    p.add_argument("--tick-fontsize", type=int, default=24,
                   help="Axis tick font size")
    p.add_argument("--legend-fontsize", type=int, default=20,
                   help="Legend font size")

    p.add_argument("--fig-width", type=float, default=9.0,
                   help="Figure width in inches")
    p.add_argument("--fig-height", type=float, default=8.0,
                   help="Figure height in inches")

    p.add_argument("--legend-loc", type=str, default="upper right",
                   choices=[
                       "upper right", "upper left", "lower right", "lower left",
                       "center right", "center left", "upper center",
                       "lower center", "best"
                   ],
                   help="Legend location inside the axes")
    p.add_argument("--legend-framealpha", type=float, default=0.92,
                   help="Legend background opacity")

    return p.parse_args()


def load_activations_multi(path, layer, max_samples, include_types=None, exclude_types=None):
    buckets = {}
    counts = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading activations"), start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON at line {line_num}")
                continue

            if obj.get("layer", None) != layer:
                continue

            original_t = obj.get("type", None)
            if not isinstance(original_t, str) or not original_t.strip():
                continue

            original_t_norm = normalize_type(original_t)

            if include_types is not None and original_t_norm not in include_types:
                continue
            if exclude_types is not None and original_t_norm in exclude_types:
                continue

            final_t = remap_type(original_t_norm)

            if counts.get(final_t, 0) >= max_samples:
                continue

            act = obj.get("activation", None)
            if not isinstance(act, list) or len(act) == 0:
                continue

            try:
                act = np.asarray(act, dtype=np.float32)
            except Exception:
                continue

            if act.ndim != 1 or act.size == 0:
                continue

            buckets.setdefault(final_t, []).append(act)
            counts[final_t] = counts.get(final_t, 0) + 1

    types_sorted = stable_order_types(list(buckets.keys()))
    X_list = []
    label_list = []

    for t in types_sorted:
        X_t = np.vstack(buckets[t]).astype(np.float32)
        X_list.append(X_t)
        label_list += [t] * len(X_t)

    if len(X_list) == 0:
        return None, None, counts

    dims = [x.shape[1] for x in X_list]
    if len(set(dims)) != 1:
        raise ValueError(f"Dimension mismatch across types: {dims}")

    X_all = np.vstack(X_list).astype(np.float32)
    labels = np.array(label_list)

    return X_all, labels, counts


def plot_2d(
    X_2d,
    labels,
    output,
    equal_aspect=False,
    point_size=10.0,
    alpha=0.75,
    xlabel_fontsize=30,
    ylabel_fontsize=30,
    tick_fontsize=24,
    legend_fontsize=15,
    fig_width=9.0,
    fig_height=8.0,
    legend_loc="upper right",
    legend_framealpha=0.92,
):
    uniq = stable_order_types(labels.tolist())

    legend_handles = [
        Patch(facecolor=CATEGORY_COLORS.get(t, "#000000"), edgecolor="none") for t in uniq
    ]
    legend_labels = uniq

    base = re.sub(r"\.png$", "", output, flags=re.IGNORECASE)
    base = re.sub(r"\.pdf$", "", base, flags=re.IGNORECASE)

    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))

    for t in uniq:
        m = (labels == t)
        if not m.any():
            continue

        ax1.scatter(
            X_2d[m, 0],
            X_2d[m, 1],
            s=point_size,
            alpha=alpha,
            color=CATEGORY_COLORS.get(t, "#000000"),
            edgecolors="none",
            rasterized=True
        )

    ax1.set_xlabel("tSNE-1", fontsize=xlabel_fontsize)
    ax1.set_ylabel("tSNE-2", fontsize=ylabel_fontsize)
    ax1.tick_params(axis="both", labelsize=tick_fontsize)
    ax1.grid(True, alpha=0.3)

    if equal_aspect:
        ax1.set_aspect("equal", adjustable="box")

    legend = ax1.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc=legend_loc,
        frameon=True,
        fancybox=False,
        framealpha=legend_framealpha,
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=1.4,
        handleheight=0.8,
        fontsize=legend_fontsize
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("lightgray")

    fig1.tight_layout()

    with_legend_png = f"{base}.png"
    with_legend_pdf = f"{base}.pdf"

    fig1.savefig(with_legend_png, dpi=300)
    fig1.savefig(with_legend_pdf)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))

    for t in uniq:
        m = (labels == t)
        if not m.any():
            continue

        ax2.scatter(
            X_2d[m, 0],
            X_2d[m, 1],
            s=point_size,
            alpha=alpha,
            color=CATEGORY_COLORS.get(t, "#000000"),
            edgecolors="none",
            rasterized=True
        )

    ax2.set_xlabel("tSNE-1", fontsize=xlabel_fontsize)
    ax2.set_ylabel("tSNE-2", fontsize=ylabel_fontsize)
    ax2.tick_params(axis="both", labelsize=tick_fontsize)
    ax2.grid(True, alpha=0.3)

    if equal_aspect:
        ax2.set_aspect("equal", adjustable="box")

    fig2.tight_layout()

    no_legend_png = f"{base}_no_legend.png"
    no_legend_pdf = f"{base}_no_legend.pdf"

    fig2.savefig(no_legend_png, dpi=300)
    fig2.savefig(no_legend_pdf)
    plt.close(fig2)

    print(f"Saved figure with legend to {with_legend_png}")
    print(f"Saved figure with legend to {with_legend_pdf}")
    print(f"Saved figure without legend to {no_legend_png}")
    print(f"Saved figure without legend to {no_legend_pdf}")


def make_tsne(perplexity, learning_rate, init, metric, random_state, n_iter):
    try:
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            init=init,
            metric=metric,
            random_state=random_state,
            max_iter=n_iter,
            verbose=1
        )
    except TypeError:
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            init=init,
            metric=metric,
            random_state=random_state,
            n_iter=n_iter,
            verbose=1
        )


def main():
    args = parse_args()

    include_types = None
    exclude_types = None

    if args.include_types:
        include_types = {
            normalize_type(x) for x in args.include_types.split(",") if x.strip()
        }

    if args.exclude_types:
        exclude_types = {
            normalize_type(x) for x in args.exclude_types.split(",") if x.strip()
        }

    inferred_model = infer_model_name_from_input_path(args.input)
    model_name = sanitize_name(args.model if args.model else inferred_model)

    save_dir = os.path.join(args.outdir, model_name)
    ensure_dir(save_dir)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.no_timestamp:
        out_name = f"tsne_{model_name}_layer{args.layer}"
    else:
        out_name = f"tsne_{model_name}_layer{args.layer}_{ts}"

    output_path = os.path.join(save_dir, out_name)

    print(f"Input JSONL : {args.input}")
    print(f"Model name  : {model_name}")
    print(f"Layer       : {args.layer}")
    print(f"Figure dir  : {save_dir}")
    print(f"Figure base : {output_path}")

    print(f"\nLoading layer {args.layer} activations ...")
    X_all, labels, counts = load_activations_multi(
        args.input,
        layer=args.layer,
        max_samples=args.max_samples,
        include_types=include_types,
        exclude_types=exclude_types
    )

    print("\nCounts per final plotted type:")
    for k in stable_order_types(list(counts.keys())):
        print(f"  {k:>14s}: {counts.get(k, 0)}")

    if X_all is None or labels is None:
        print("\nNo samples found for the requested layer/types. Exiting.")
        return

    print(f"\nTotal samples: {X_all.shape[0]}, dim={X_all.shape[1]}")
    uniq_types = stable_order_types(labels.tolist())
    print(f"Final plotted classes: {uniq_types}")

    if args.center:
        X_all = X_all - X_all.mean(axis=0, keepdims=True)

    X_for_tsne = X_all
    if args.pca_dim and args.pca_dim > 0 and args.pca_dim < X_all.shape[1]:
        print(f"\nRunning PCA -> {args.pca_dim} dims ...")
        pca = PCA(n_components=args.pca_dim, random_state=args.random_state)
        X_for_tsne = pca.fit_transform(X_all).astype(np.float32)
        print(f"PCA done: shape={X_for_tsne.shape}")
    else:
        print("\nSkipping PCA ...")

    n = X_for_tsne.shape[0]
    max_perp = (n - 1) / 3.0

    perp = float(args.perplexity)
    if not np.isfinite(perp) or perp <= 0:
        print(f"\n[WARN] Invalid perplexity={args.perplexity}. Falling back to 30.0")
        perp = 30.0

    if n < 3:
        print("\n[WARN] Too few samples for tSNE. Need at least 3 samples.")
        return

    if perp >= max_perp:
        perp = max(5.0, min(30.0, max_perp - 1.0))
        if perp <= 0:
            perp = 5.0
        print(f"\n[WARN] perplexity too large for N={n}. Using perplexity={perp:.2f}")

    lr = args.learning_rate
    if lr != "auto":
        try:
            lr = float(lr)
        except ValueError:
            print(f"\n[WARN] Invalid --learning-rate '{args.learning_rate}', using 'auto'.")
            lr = "auto"

    print("\nRunning tSNE ...")
    tsne = make_tsne(
        perplexity=perp,
        learning_rate=lr,
        init=args.init,
        metric=args.metric,
        random_state=args.random_state,
        n_iter=args.n_iter
    )

    X_2d = tsne.fit_transform(X_for_tsne).astype(np.float32)

    plot_2d(
        X_2d=X_2d,
        labels=labels,
        output=output_path,
        equal_aspect=args.equal_aspect,
        point_size=args.point_size,
        alpha=args.alpha,
        xlabel_fontsize=args.xlabel_fontsize,
        ylabel_fontsize=args.ylabel_fontsize,
        tick_fontsize=args.tick_fontsize,
        legend_fontsize=args.legend_fontsize,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        legend_loc=args.legend_loc,
        legend_framealpha=args.legend_framealpha,
    )


if __name__ == "__main__":
    main()