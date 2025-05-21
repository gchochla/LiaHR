import argparse
import os
import yaml
import re
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score, f1_score


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

    gt_results = {}  # random query results wrt GT
    results = {}
    order = {}

    gt_indexed_metrics_glob = os.path.join(
        args.gt[0].replace("{}", "*"), "indexed_metrics.yml"
    )
    with open(glob(gt_indexed_metrics_glob)[0]) as fp:
        metrics = yaml.safe_load(fp)
        gt = {
            k: v.get("test_true", v.get("test_gt", None))
            for k, v in metrics["experiment_0"].items()
            if k != "description"
        }
    mlb = MultiLabelBinarizer()
    mlb.fit(list(gt.values()))
    gt = {k: mlb.transform([v])[0] for k, v in gt.items()}

    for dir in args.gt + args.random:
        fn_glob = os.path.join(dir.replace("{}", "*"), "aggregated_metrics.yml")
        fn_re = os.path.join(
            dir.replace("{}", "(\\d+)"), "aggregated_metrics.yml"
        )
        order_re = re.compile(fn_re)
        for fn in glob(fn_glob):
            with open(fn) as fp:
                gt_metrics = yaml.safe_load(fp)
            js = gt_metrics[""][f"test_{args.metric}"]
            mean, std = js.split("+-")
            results.setdefault(dir, []).append((float(mean), float(std)))
            order.setdefault(dir, []).append(
                int(order_re.match(fn).groups()[0])
            )

        inds = np.argsort(order[dir])
        order[dir] = np.array(order[dir])[inds]
        results[dir] = np.array(results[dir])[inds]

        if dir in args.random:
            indexed_metrics_glob = os.path.join(
                dir.replace("{}", "*"), "indexed_metrics.yml"
            )
            indexed_metrics_re = re.compile(
                os.path.join(dir.replace("{}", "(\\d+)"), "indexed_metrics.yml")
            )
            matched_shots = []
            for fn in glob(indexed_metrics_glob):
                matched_shots.append(
                    int(indexed_metrics_re.match(fn).groups()[0])
                )
                with open(fn) as fp:
                    exp_metrics = yaml.safe_load(fp)

                fn_metrics = []

                for _, metrics in exp_metrics.items():
                    preds = {
                        k: v.get("test_pred", v.get("test_preds", None))
                        for k, v in metrics.items()
                        if k != "description"
                    }
                    preds = {k: mlb.transform([preds[k]])[0] for k in gt}

                    if args.metric == "jaccard_score":
                        metric = jaccard_score(
                            np.array(list(gt.values())),
                            np.array(list(preds.values())),
                            average="samples",
                            zero_division=1,
                        )
                    elif args.metric == "macro_f1":
                        metric = f1_score(
                            np.array(list(gt.values())),
                            np.array(list(preds.values())),
                            average="macro",
                            zero_division=0,
                        )
                    elif args.metric == "micro_f1":
                        metric = f1_score(
                            np.array(list(gt.values())),
                            np.array(list(preds.values())),
                            average="micro",
                            zero_division=0,
                        )
                    fn_metrics.append(metric)

                gt_results.setdefault(dir, []).append(
                    (np.mean(fn_metrics), np.std(fn_metrics))
                )

            inds = np.argsort(matched_shots)
            gt_results[dir] = np.array(gt_results[dir])[inds]

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
    fig_sub, ax_sub = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=(ncols * 3, nrows * 2.5),
    )

    fig_all, ax_all = plt.subplots()

    for i, (gt, random) in enumerate(zip(args.gt, args.random)):
        ax_all.plot(
            order[gt],
            100
            * (results[gt][:, 0] - results[random][:, 0])
            / results[gt][:, 0],
            label=args.experiment_names[i],
            linestyle="--",
            marker="x",
            color=colors(family_mapping[i][0]),
            alpha=(family_mapping[i][1]) * (-0.99) + 1,
        )

        ax_sub[family_inds[i]].scatter(
            order[gt],
            results[gt][:, 0],
            color="dodgerblue",
            label="Query w/ gold",
            s=35,
            marker="o",
        )
        ax_sub[family_inds[i]].errorbar(
            order[gt],
            results[gt][:, 0],
            yerr=results[gt][:, 1],
            color="dodgerblue",
            linestyle="None",
            capsize=2,
        )
        ax_sub[family_inds[i]].scatter(
            order[random],
            results[random][:, 0],
            color="orange",
            label="Query w/ rand",
            s=35,
            marker="d",
        )
        ax_sub[family_inds[i]].errorbar(
            order[random],
            results[random][:, 0],
            yerr=results[random][:, 1],
            color="orange",
            linestyle="None",
            capsize=2,
        )
        ax_sub[family_inds[i]].scatter(
            order[random],
            gt_results[random][:, 0],
            color="gray",
            label="Gold perf w/ rand",
            s=35,
            marker="s",
        )
        ax_sub[family_inds[i]].errorbar(
            order[random],
            gt_results[random][:, 0],
            yerr=gt_results[random][:, 1],
            color="gray",
            linestyle="None",
            capsize=2,
        )
        ax_sub[family_inds[i]].set_ylim(0, 1.02)

        ax_sub[family_inds[i]].set_title(args.experiment_names[i])
        ax_sub[family_inds[i]].set_xticks(order[gt])

        if not isinstance(family_inds[i], int):
            if family_inds[i][0] == nrows - 1:
                ax_sub[family_inds[i]].set_xlabel("Order")

        fig_single, ax_single = plt.subplots()

        # plot performance bars + stds for each model for both gt and random
        # for each shot, two bars for gt and random should appear side by side

        width = 0.5

        ax_single.bar(
            order[gt] - width,
            100 * results[gt][:, 0],
            yerr=100 * results[gt][:, 1],
            width=width,
            color="orange",
            label="GT query",
            edgecolor="black",
        )
        ax_single.bar(
            order[random],
            100 * results[random][:, 0],
            yerr=100 * results[random][:, 1],
            width=width,
            color="cyan",
            label="Random query",
            edgecolor="black",
        )
        ax_single.bar(
            order[random] + width,
            100 * gt_results[random][:, 0],
            yerr=100 * gt_results[random][:, 1],
            width=width,
            color="limegreen",
            label="Random query\n(wrt GT)",
            edgecolor="black",
        )

        ax_single.set_title(
            f"{args.experiment_names[i]} {args.metric.replace('_', ' ').title()}"
        )

        ax_single.set_xlabel("Query Position")
        ax_single.set_ylabel(args.metric.replace("_", " ").title())
        ax_single.set_xticks(order[gt])
        ax_single.legend(loc="upper center")
        ax_single.set_ylim(
            min(
                results[random][:, 0].min(),
                results[gt][:, 0].min(),
                gt_results[random][:, 0].min(),
            )
            * 95,
            max(
                results[random][:, 0].max(),
                results[gt][:, 0].max(),
                gt_results[random][:, 0].max(),
            )
            * 105,
        )

        fig_single.savefig(
            os.path.join(
                args.output_folder,
                f"{args.experiment_names[i]}-{args.metric}-order.png",
            )
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
                            label="Query w/ gold",
                            color="dodgerblue",
                            marker="o",
                        )
                        ax_sub[i, j].scatter(
                            [None],
                            [None],
                            label="Query w/ rand",
                            color="orange",
                            marker="d",
                        )
                        ax_sub[i, j].scatter(
                            [None],
                            [None],
                            label="Gold perf w/ rand",
                            color="gray",
                            marker="s",
                        )
                        ax_sub[i, j].legend(loc="upper center")
                        legend = True
    if not legend:
        ax_sub[family_inds[0]].legend(loc="center")

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
    fig_sub.supylabel(args.metric.replace("_", " ").title(), fontsize=14)

    # set x title at the middle of the bottom row
    if isinstance(family_inds[0], int):
        fig_sub.text(
            0.54,
            0.02,
            "Query Order in Prompt",
            ha="center",
            va="center",
            fontsize=14,
        )
    fig_sub.tight_layout()
    fig_sub.savefig(
        os.path.join(args.output_folder, f"{args.metric}-order-box.png"),
        bbox_inches="tight",
    )
    fig_sub.savefig(
        os.path.join(args.output_folder, f"{args.metric}-order-box.pdf"),
        bbox_inches="tight",
    )

    ax_all.set_title(f"{args.metric.replace('_', ' ').title()} degradation")
    ax_all.set_xlabel("Query Order")
    ax_all.set_ylabel("Relative degradation")
    ax_all.set_xticks(sorted(set.union(*[set(order[gt]) for gt in args.gt])))
    ax_all.legend()
    ax_all.grid()
    ax_all.yaxis.set_major_formatter(mtick.PercentFormatter())

    fig_all.savefig(
        os.path.join(args.output_folder, f"{args.metric}-degradation-order.png")
    )


if __name__ == "__main__":
    main()
