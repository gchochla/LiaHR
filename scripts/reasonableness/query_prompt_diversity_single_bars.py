import argparse
import os
import yaml
from glob import glob

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt",
        nargs="+",
        type=str,
        required=True,
        help="folders with GT query label",
    )
    parser.add_argument(
        "--annotator",
        nargs="+",
        type=str,
        required=True,
        help="folders with annotators query label, corresponding to GT folders",
    )
    parser.add_argument(
        "--random",
        nargs="+",
        type=str,
        required=True,
        help="folders with random query label, corresponding to GT folders",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="jaccard_score",
        help="metric to compare",
    )
    parser.add_argument(
        "--experiment-names",
        nargs="+",
        type=str,
        required=True,
        help="names of the experiments",
    )
    parser.add_argument(
        "--family-names",
        nargs="+",
        type=str,
        required=True,
        help="names of the model families",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="output folder to save the plot",
    )
    parser.add_argument(
        "--model-families",
        nargs="+",
        type=str,
        help="range of models that are in the same model family for coloring",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output figure",
    )

    args = parser.parse_args()

    assert len(args.gt) == len(args.random)

    return args


def main():
    args = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    results = {}

    for dir in args.gt + args.random:
        fn = os.path.join(dir, "aggregated_metrics.yml")
        with open(fn) as fp:
            gt_metrics = yaml.safe_load(fp)
        js = gt_metrics[""][f"test_{args.metric}"]
        mean, std = js.split("+-")
        results[dir] = float(mean), float(std)

    ann_results = {}
    for dir in args.annotator:
        fn_glob = os.path.join(
            dir.replace("{}", "*"), "aggregated_annotator_metrics.yml"
        )
        for fn in glob(fn_glob):
            with open(fn) as fp:
                gt_metrics = yaml.safe_load(fp)

            for ann in gt_metrics[""]:
                metric = gt_metrics[""][ann][f"{args.metric}"]
                mean, std = metric.split("+-")
                ann_results.setdefault(dir, {})[ann] = float(mean), float(std)

    if args.model_families:
        model_families = [
            list(map(int, e.split("-"))) for e in args.model_families
        ]
        model_families = [
            (e[0], e[1] + 1) if len(e) == 2 else (e[0], e[0] + 1)
            for e in model_families
        ]
        model_families = sorted(model_families, key=lambda e: e[0])
    else:
        model_families = [(i, i + 1) for i in range(len(args.gt))]

    nrows = len(args.family_names)

    # Create figure with slightly improved aesthetics but keeping original style
    fig, axs = plt.subplots(
        nrows=nrows, ncols=1, sharex=True, sharey=True, figsize=(6, 8)
    )

    ann_colors = ["red", "teal", "gold"]

    # Handle case of single subplot
    if nrows == 1:
        axs = [axs]

    for i, (gt_fn, random_fn, ann_fn) in enumerate(
        zip(args.gt, args.random, args.annotator)
    ):
        row_idx = i % len(args.family_names)
        col_idx = i // len(args.family_names)
        width = 1 / (len(ann_results[ann_fn]) + 2) / 1.3

        label_cond = row_idx == len(args.family_names) - 1 and col_idx == 0

        for j, (ann, ann_res) in enumerate(ann_results[ann_fn].items()):
            axs[row_idx].bar(
                col_idx - (len(ann_results[ann_fn]) - j - 1) * width,
                ann_res[0],
                yerr=ann_res[1],
                width=width,
                color=ann_colors[j],
                label=(f"Ann{j}" if label_cond else None),
                edgecolor="black",
            )

        if label_cond:
            # do it only once for the final plot
            aggr = "Aggregate"
            random = "Random"
        else:
            aggr = random = None

        axs[row_idx].bar(
            col_idx + width,
            results[gt_fn][0],
            yerr=results[gt_fn][1],
            width=width,
            color="dodgerblue",
            label=aggr,
            edgecolor="black",
        )

        axs[row_idx].bar(
            col_idx + 2 * width,
            results[random_fn][0],
            yerr=results[random_fn][1],
            width=width,
            color="orange",
            label=random,
            edgecolor="black",
        )

        axs[row_idx].set_title(
            args.family_names[row_idx], fontsize=14, fontweight='bold'
        )

    # Improved y-axis label but keeping it simple
    fig.supylabel("Success Rate", fontsize=14)
    axs[-1].legend(frameon=True, fontsize=12, loc="lower right")

    for ax in axs[:-1]:
        # disable x-ticks for all but the last subplot
        ax.set_xticks(
            [-1 / 2]
            + list(range(len(args.experiment_names)))
            + [len(args.experiment_names) - 1 / 2]
        )
        ax.set_xticklabels([None] * (len(args.experiment_names) + 2))

    # Keep original x-ticks with margin space
    axs[-1].set_xticks(
        [-1 / 2]
        + list(range(len(args.experiment_names)))
        + [len(args.experiment_names) - 1 / 2]
    )
    axs[-1].set_xticklabels(
        [None] + args.experiment_names + [None], rotation=45, fontsize=12
    )

    for ax in axs:
        # Add very subtle grid for readability
        ax.grid(axis='y', linestyle=':', alpha=0.2)

    # Use tight_layout for better spacing
    fig.tight_layout()

    # Save with higher DPI
    fig.savefig(
        os.path.join(
            args.output_folder, f"{args.metric}-diversity-single-bars.png"
        ),
        dpi=args.dpi,
        bbox_inches='tight',
    )

    # Also save as PDF for publication
    fig.savefig(
        os.path.join(
            args.output_folder, f"{args.metric}-diversity-single-bars.pdf"
        ),
        dpi=args.dpi,
        bbox_inches='tight',
    )

    print(f"Plot saved to {args.output_folder}")


if __name__ == "__main__":
    main()
