import argparse
import os
import yaml
import re
from glob import glob
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ing",
        nargs="+",
        type=str,
        required=True,
        help="folders with GT query label for in-group",
    )
    parser.add_argument(
        "--outg",
        nargs="+",
        type=str,
        required=True,
        help="folders with GT query label for out-group",
    )
    parser.add_argument(
        "--random-ing",
        nargs="+",
        type=str,
        required=True,
        help="folders with random query label, corresponding to GT folders for in-group",
    )
    parser.add_argument(
        "--random-outg",
        nargs="+",
        type=str,
        required=True,
        help="folders with random query label, corresponding to GT folders for out-group",
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

    assert (
        len(args.ing)
        == len(args.outg)
        == len(args.random_ing)
        == len(args.random_outg)
    ), "Number of folders for GT and random query labels should be the same"

    return args


def main():
    args = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    results = {}
    shots = {}

    for dir in args.ing + args.outg + args.random_ing + args.random_outg:
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
        model_families = [(i, i + 1) for i in range(len(args.ing))]
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
    fig_sub, ax_sub = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=(ncols * 3, nrows * 2.5),
    )
    fig_all_in, ax_all_in = plt.subplots()
    fig_all_out, ax_all_out = plt.subplots()

    for i, (ing, outg, randoming, randomoutg) in enumerate(
        zip(args.ing, args.outg, args.random_ing, args.random_outg)
    ):
        print(f"Processing {args.experiment_names[i]}...")
        ax_all_in.plot(
            shots[ing],
            100
            * (results[ing][:, 0] - results[randoming][:, 0])
            / results[ing][:, 0],
            label=args.experiment_names[i],
            linestyle="--",
            marker="x",
            color=colors(family_mapping[i][0]),
            alpha=(family_mapping[i][1]) * (-0.99) + 1,
        )
        ax_all_out.plot(
            shots[outg],
            100
            * (results[outg][:, 0] - results[randomoutg][:, 0])
            / results[outg][:, 0],
            label=args.experiment_names[i],
            linestyle="--",
            marker="x",
            color=colors(family_mapping[i][0]),
            alpha=(family_mapping[i][1]) * (-0.99) + 1,
        )

        # plot performance bars + stds for each model for both gt and random
        # for each shot, two bars for gt and random should appear side by side

        width = 2

        effective_shots = deepcopy(shots[ing])
        effective_shots[effective_shots > 30] = (
            effective_shots[effective_shots > 30] * 0.8
        )

        ax_sub[family_inds[i]].bar(
            effective_shots - 2 * width,
            results[ing][:, 0],
            yerr=results[ing][:, 1],
            width=width,
            color="dodgerblue",
            # label="IN-group",
            edgecolor="black",
        )
        ax_sub[family_inds[i]].bar(
            effective_shots - width,
            results[randoming][:, 0],
            yerr=results[randoming][:, 1],
            width=width,
            color="dodgerblue",
            # label="Random IN-group",
            edgecolor="black",
            hatch="\\\\",
        )
        ax_sub[family_inds[i]].bar(
            effective_shots,
            results[outg][:, 0],
            yerr=results[outg][:, 1],
            width=width,
            color="orange",
            # label="OUT-group",
            edgecolor="black",
        )
        ax_sub[family_inds[i]].bar(
            effective_shots + width,
            results[randomoutg][:, 0],
            yerr=results[randomoutg][:, 1],
            width=width,
            color="orange",
            # label="Random OUT-group",
            edgecolor="black",
            hatch="\\\\",
        )

        ax_sub[family_inds[i]].set_ylim(0.4, 1)

        ax_sub[family_inds[i]].set_title(args.experiment_names[i])
        ax_sub[family_inds[i]].set_xticks(effective_shots)
        ax_sub[family_inds[i]].set_xticklabels(shots[ing])

        if not isinstance(family_inds[i], int):
            if family_inds[i][0] == nrows - 1:
                ax_sub[family_inds[i]].set_xlabel("Shots")

    all_title = f"{args.metric.replace('_', ' ').title()} degradation"
    if args.dataset_name:
        all_title += f" on {args.dataset_name}"

    ax_all_in.set_title(all_title + " (IN-group)")
    ax_all_in.set_xlabel("Shots")
    ax_all_in.set_ylabel("Relative degradation")
    ax_all_in.set_xticks(sorted(set.union(*[set(shots[i]) for i in args.ing])))
    ax_all_in.legend()
    ax_all_in.grid()
    ax_all_in.yaxis.set_major_formatter(mtick.PercentFormatter())

    fig_all_in.savefig(
        os.path.join(args.output_folder, f"{args.metric}-degradation-in.png")
    )
    fig_all_in.savefig(
        os.path.join(args.output_folder, f"{args.metric}-degradation-in.pdf"),
        bbox_inches="tight",
    )

    ax_all_out.set_title(all_title + " (OUT-group)")
    ax_all_out.set_xlabel("Shots")
    ax_all_out.set_ylabel("Relative degradation")
    ax_all_out.set_xticks(sorted(set.union(*[set(shots[i]) for i in args.ing])))
    ax_all_out.legend()
    ax_all_out.grid()
    ax_all_out.yaxis.set_major_formatter(mtick.PercentFormatter())
    fig_all_out.savefig(
        os.path.join(args.output_folder, f"{args.metric}-degradation-out.png")
    )
    fig_all_out.savefig(
        os.path.join(args.output_folder, f"{args.metric}-degradation-out.pdf"),
        bbox_inches="tight",
    )

    legend = False
    if nrows > 1:
        for i in range(nrows):
            for j in range(ncols):
                if j >= family_model_cnt[i]:
                    ax_sub[i, j].axis("off")
                    if j == ncols - 1 and not legend:
                        ax_sub[i, j].bar(
                            x=np.nan,
                            height=np.nan,
                            label="IN-group",
                            color="dodgerblue",
                        )
                        ax_sub[i, j].bar(
                            x=np.nan,
                            height=np.nan,
                            label="OUT-group",
                            color="orange",
                        )
                        ax_sub[i, j].bar(
                            x=np.nan,
                            height=np.nan,
                            label="Opposite",
                            color="gray",
                            hatch="\\\\",
                        )
                        ax_sub[i, j].legend(loc="upper center")
                        legend = True
    if not legend:
        ax_sub[family_inds[len(args.gt) - 1]].bar(
            [None], [None], label="IN-group", color="dodgerblue"
        )
        ax_sub[family_inds[len(args.gt) - 1]].bar(
            [None], [None], label="OUT-group", color="orange"
        )
        ax_sub[family_inds[len(args.gt) - 1]].bar(
            [None], [None], label="Opposite", color="black", hatch="\\"
        )
        ax_sub[family_inds[len(args.gt) - 1]].legend()

    if args.dataset_name:
        # Normal prefix, dataset name in bold + sans‑serif
        # Bold the dataset name and preserve spaces/underscores inside mathtext
        ds_math = args.dataset_name.replace(" ", r"\ ").replace("_", r"\_")
        # change minus to regular dash in math env
        text = "Label in a Haystack".replace(" ", r"\text{-}")
        sub_title = rf"${text}$ success rate on $\bf{{{ds_math}}}$"
    else:
        sub_title = ""
    fig_sub.suptitle(sub_title, fontsize=14)
    fig_sub.supylabel(
        (
            args.metric.replace("_", " ").title()
            if "auc" not in args.metric
            else "ROC-AUC"
        ),
        fontsize=14,
    )

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
    fig_sub.savefig(os.path.join(args.output_folder, f"{args.metric}-bars.png"))
    fig_sub.savefig(
        os.path.join(args.output_folder, f"{args.metric}-bars.pdf"),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
