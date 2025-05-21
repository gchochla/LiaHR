import argparse
import os
import yaml
import re
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


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
        "--random",
        nargs="+",
        type=str,
        required=True,
        help="folders with random query label, corresponding to GT folders",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
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
        "--dataset-name",
        type=str,
        help="dataset name to add to the plot title",
    )

    args = parser.parse_args()

    assert len(args.gt) == len(args.random)

    return args


def main():
    args = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    results = {}
    shots = {}

    for dir in args.gt + args.random:
        fn_glob = os.path.join(dir.replace("{}", "*"), "aggregated_metrics.yml")
        fn_re = os.path.join(
            dir.replace("{}", "(\\d+)"), "aggregated_metrics.yml"
        )
        shots_re = re.compile(fn_re)
        for fn in glob(fn_glob):
            with open(fn) as fp:
                gt_metrics = yaml.safe_load(fp)
            js = gt_metrics[""][f"test_{args.metric}"]
            mean, std = js.split("+-")
            results.setdefault(dir, []).append((float(mean), float(std)))
            shots.setdefault(dir, []).append(
                int(shots_re.match(fn).groups()[0])
            )

        inds = np.argsort(shots[dir])
        shots[dir] = np.array(shots[dir])[inds]
        results[dir] = np.array(results[dir])[inds]

    if args.model_families:
        model_families = [
            list(map(int, e.split("-"))) for e in args.model_families
        ]
        model_families = [
            (e[0], e[1] + 1) if len(e) == 2 else (e[0], e[0] + 1)
            for e in model_families
        ]
        model_families = sorted(model_families, key=lambda e: e[0])
        family_mapping = {
            model: (i, (model - family[0]) / (family[1] - family[0]))
            for i, family in enumerate(model_families)
            for model in range(family[0], family[1])
        }
        max_family = max([family[1] - family[0] for family in model_families])
        family_inds = {
            model: (
                (i, model - family[0])
                if max_family > 1 and len(model_families) > 1
                else (model - family[0]) if len(model_families) == 1 else i
            )
            for i, family in enumerate(model_families)
            for model in range(family[0], family[1])
        }
        family_model_cnt = {
            i: family[1] - family[0] for i, family in enumerate(model_families)
        }
    else:
        model_families = [(i, i + 1) for i in range(len(args.gt))]
        family_mapping = {
            model[0]: (i, 0) for i, model in enumerate(model_families)
        }
        family_inds = {model[0]: i for i, model in enumerate(model_families)}
        family_model_cnt = {model[0]: 1 for model in model_families}

    colors = plt.cm.get_cmap("Set1", len(model_families))

    nrows = len(model_families)
    ncols = max([family[1] - family[0] for family in model_families])
    if ncols == 1:
        nrows, ncols = ncols, nrows
    fig_all, ax_all = plt.subplots()
    fig_sub, ax_sub = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=(ncols * 3, nrows * 2.5),
    )

    for i, (gt, random) in enumerate(zip(args.gt, args.random)):
        ax_all.plot(
            shots[gt],
            100
            * (results[gt][:, 0] - results[random][:, 0])
            / results[gt][:, 0],
            label=args.experiment_names[i],
            linestyle="--",
            marker="x",
            color=colors(family_mapping[i][0]),
            alpha=(family_mapping[i][1]) * (-0.99) + 1,
        )

        fig_single, ax_single = plt.subplots()

        # plot performance bars + stds for each model for both gt and random
        # for each shot, two bars for gt and random should appear side by side

        width = 2

        ax_sub[family_inds[i]].scatter(
            shots[gt],
            results[gt][:, 0],
            color="dodgerblue",
            label="Gold pair",
            s=35,
            marker="o",
        )
        ax_sub[family_inds[i]].errorbar(
            shots[gt],
            results[gt][:, 0],
            yerr=results[gt][:, 1],
            color="dodgerblue",
            linestyle="None",
            capsize=2,
        )
        ax_sub[family_inds[i]].scatter(
            shots[random],
            results[random][:, 0],
            color="orange",
            label="Rand pair",
            s=35,
            marker="d",
        )
        ax_sub[family_inds[i]].errorbar(
            shots[random],
            results[random][:, 0],
            yerr=results[random][:, 1],
            color="orange",
            linestyle="None",
            capsize=2,
        )
        ax_sub[family_inds[i]].set_ylim(0, 1)

        ax_sub[family_inds[i]].set_title(args.experiment_names[i])
        ax_sub[family_inds[i]].set_xticks(shots[gt])

        if not isinstance(family_inds[i], int):
            if family_inds[i][0] == nrows - 1:
                ax_sub[family_inds[i]].set_xlabel("Shots")

        ax_single.bar(
            shots[gt] - width / 2,
            100 * results[gt][:, 0],
            yerr=100 * results[gt][:, 1],
            width=width,
            color="orange",
            label="Gold pair",
            edgecolor="black",
        )
        ax_single.bar(
            shots[random] + width / 2,
            100 * results[random][:, 0],
            yerr=100 * results[random][:, 1],
            width=width,
            color="cyan",
            label="Rand pair",
            edgecolor="black",
        )

        single_title = f"{args.experiment_names[i]} {args.metric.replace('_', ' ').title()}"
        if args.dataset_name:
            single_title += f" on {args.dataset_name}"
        ax_single.set_title(single_title)

        ax_single.set_xlabel("Shots")
        ax_single.set_ylabel(args.metric.replace("_", " ").title())
        ax_single.set_xticks(shots[gt])
        ax_single.legend(loc="upper center")
        ax_single.set_ylim(
            min(results[random][:, 0].min(), results[gt][:, 0].min()) * 90,
            max(results[random][:, 0].max(), results[gt][:, 0].max()) * 110,
        )

        fig_single.savefig(
            os.path.join(
                args.output_folder,
                f"baseline-{args.experiment_names[i]}-{args.metric}.png",
            )
        )
        fig_single.savefig(
            os.path.join(
                args.output_folder,
                f"baseline-{args.experiment_names[i]}-{args.metric}.pdf",
            ),
            bbox_inches="tight",
        )

    all_title = f"{args.metric.replace('_', ' ').title()} degradation"
    if args.dataset_name:
        all_title += f" on {args.dataset_name}"

    ax_all.set_title(all_title)
    ax_all.set_xlabel("Shots")
    ax_all.set_ylabel("Relative degradation")
    ax_all.set_xticks(sorted(set.union(*[set(shots[gt]) for gt in args.gt])))
    ax_all.legend()
    ax_all.grid()
    ax_all.yaxis.set_major_formatter(mtick.PercentFormatter())

    fig_all.savefig(
        os.path.join(
            args.output_folder, f"baseline-{args.metric}-degradation.png"
        )
    )
    fig_all.savefig(
        os.path.join(
            args.output_folder, f"baseline-{args.metric}-degradation.pdf"
        ),
        bbox_inches="tight",
    )

    legend = False
    if nrows > 1:
        for i in range(nrows):
            for j in range(ncols):
                if j >= family_model_cnt[i]:
                    ax_sub[i, j].axis("off")
                    if j == ncols - 1 and not legend:
                        ax_sub[i, j].scatter(
                            [None],
                            [None],
                            label="Gold pair",
                            color="dodgerblue",
                            marker="o",
                        )
                        ax_sub[i, j].scatter(
                            [None],
                            [None],
                            label="Rand pair",
                            color="orange",
                            marker="d",
                        )
                        ax_sub[i, j].legend(loc="upper center")
                        legend = True
    if not legend:
        ax_sub[family_inds[len(args.gt) - 1]].legend()

    if args.dataset_name:
        # Normal prefix, dataset name in bold + sans‑serif
        # Bold the dataset name and preserve spaces/underscores inside mathtext
        ds_math = args.dataset_name.replace(" ", r"\ ").replace("_", r"\_")
        # change minus to regular dash in math env
        sub_title = rf"$Baseline$ success rate on $\bf{{{ds_math}}}$"
    else:
        sub_title = ""
    fig_sub.suptitle(sub_title, fontsize=16)
    fig_sub.supylabel(args.metric.replace("_", " ").title(), fontsize=14)

    # set x title at the middle of the bottom row
    if isinstance(family_inds[0], int):
        fig_sub.text(
            0.53,
            0.02,
            "Shots",
            ha="center",
            va="center",
            fontsize=14,
        )
    fig_sub.tight_layout()
    fig_sub.savefig(
        os.path.join(args.output_folder, f"baseline-{args.metric}-box.png")
    )
    fig_sub.savefig(
        os.path.join(args.output_folder, f"baseline-{args.metric}-box.pdf"),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
