import os
from glob import glob

import pandas as pd
import numpy as np
import gridparse
from scipy.stats import binomtest, chi2_contingency


def parse_args():
    parser = gridparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--experiment_dir", nargs="+")

    return parser.parse_args()


def icl_vs_gt(args):
    for experiment_dir in args.experiment_dir:
        # load more_correct and errors and their labels
        print("Experiment dir:", experiment_dir)

        more_correct_fns = os.path.join(experiment_dir, "more_correct_*.csv")
        more_correct_idx = pd.read_csv(
            os.path.join(experiment_dir, "more_correct-labels.csv")
        )["labels"].values.flatten()

        model_preferred = 0
        all = 0

        for more_correct_fn in glob(more_correct_fns):
            # -1 to bring to {0, 1} from {1, 2}
            more_correct = pd.read_csv(more_correct_fn)["best"].values - 1

            for_model = (more_correct_idx == more_correct).sum()
            model_preferred += for_model
            all += len(more_correct)

        model_percent = model_preferred / all
        print(f"\tModel preferred: {model_percent*100:.2f}%")
        res_preferred = binomtest(model_preferred, all, 0.5)

        with open(os.path.join(experiment_dir, "results.txt"), "w") as fp:
            fp.write(f"Preferred: {res_preferred}\n")
            fp.write(
                f"\tModel: {model_preferred}\n"
                f"\tGT: {all - model_preferred}\n"
                f"\tModel preferred: {model_percent*100}%\n"
            )


def query_prompt_vs_gt(args):
    for experiment_dir in args.experiment_dir:
        print("Experiment dir:", experiment_dir)
        # load more_correct and errors and their labels

        more_correct_fns = os.path.join(experiment_dir, "more_correct_*.csv")
        more_correct_idx = pd.read_csv(
            os.path.join(experiment_dir, "more_correct-labels.csv")
        )["labels"].values.flatten()

        model_preferred = 0
        all = 0

        for more_correct_fn in glob(more_correct_fns):
            # -1 to bring to {0, 1} from {1, 2}
            more_correct = pd.read_csv(more_correct_fn)["best"].values - 1

            for_model = (more_correct_idx == more_correct).sum()
            model_preferred += for_model
            all += len(more_correct)

        model_percent = model_preferred / all
        print(f"\tModel preferred: {model_percent*100:.2f}%")

        reasonable_fns = os.path.join(experiment_dir, "errors_*.csv")
        reasonable_idx = pd.read_csv(
            os.path.join(experiment_dir, "errors-labels.csv")
        )["labels"].values.flatten()

        contingency = {
            "cp right": {"reasonable": 0, "unreasonable": 0},
            "cp wrong": {"reasonable": 0, "unreasonable": 0},
        }

        for reasonable_fn in glob(reasonable_fns):
            reasonable = pd.read_csv(reasonable_fn)["reasonable"].values
            model_examples = reasonable[reasonable_idx == 0]
            gt_examples = reasonable[reasonable_idx == 1]

            contingency["cp wrong"]["reasonable"] += (model_examples == 1).sum()
            contingency["cp wrong"]["unreasonable"] += (
                model_examples == 0
            ).sum()

            contingency["cp right"]["reasonable"] += (gt_examples == 1).sum()
            contingency["cp right"]["unreasonable"] += (gt_examples == 0).sum()

        print(f"\tContingency: {contingency}")

        contingency_array = np.array(
            [
                [
                    contingency["cp right"]["reasonable"],
                    contingency["cp right"]["unreasonable"],
                ],
                [
                    contingency["cp wrong"]["reasonable"],
                    contingency["cp wrong"]["unreasonable"],
                ],
            ]
        )

        # run stat tests between model and gt

        res_preferred = binomtest(model_preferred, all, 0.5)
        res_reasonable = chi2_contingency(contingency_array)

        # store in experiment_dir

        with open(os.path.join(experiment_dir, "results.txt"), "w") as fp:
            fp.write(f"Preferred: {res_preferred}\n")
            fp.write(
                f"\tModel: {model_preferred}\n"
                f"\tGT: {all - model_preferred}\n"
                f"\tModel preferred: {model_percent*100}%\n"
            )

            fp.write(f"Reasonable: {res_reasonable}\n")
            fp.write(
                f"\tCP Right: {contingency['cp right']['reasonable']} reasonable, "
                f"{contingency['cp right']['unreasonable']} unreasonable\n"
                f"\tCP Wrong: {contingency['cp wrong']['reasonable']} reasonable, "
                f"{contingency['cp wrong']['unreasonable']} unreasonable\n"
            )


def baseline_vs_gt(args):
    for experiment_dir in args.experiment_dir:
        print("Experiment dir:", experiment_dir)
        # load more_correct and errors and their labels

        reasonable_fns = os.path.join(experiment_dir, "errors_*.csv")
        reasonable_idx = pd.read_csv(
            os.path.join(experiment_dir, "errors-labels.csv")
        )["labels"].values.flatten()

        contingency = {
            "cp right": {"reasonable": 0, "unreasonable": 0},
            "cp wrong": {"reasonable": 0, "unreasonable": 0},
        }

        for reasonable_fn in glob(reasonable_fns):
            reasonable = pd.read_csv(reasonable_fn)["reasonable"].values
            model_examples = reasonable[reasonable_idx == 0]
            gt_examples = reasonable[reasonable_idx == 1]

            contingency["cp wrong"]["reasonable"] += (model_examples == 1).sum()
            contingency["cp wrong"]["unreasonable"] += (
                model_examples == 0
            ).sum()

            contingency["cp right"]["reasonable"] += (gt_examples == 1).sum()
            contingency["cp right"]["unreasonable"] += (gt_examples == 0).sum()

        print(f"\tContingency: {contingency}")

        contingency_array = np.array(
            [
                [
                    contingency["cp right"]["reasonable"],
                    contingency["cp right"]["unreasonable"],
                ],
                [
                    contingency["cp wrong"]["reasonable"],
                    contingency["cp wrong"]["unreasonable"],
                ],
            ]
        )

        # run stat tests between model and gt

        res_reasonable = chi2_contingency(contingency_array)

        # store in experiment_dir

        with open(os.path.join(experiment_dir, "results.txt"), "w") as fp:
            fp.write(f"Reasonable: {res_reasonable}\n")
            fp.write(
                f"\tCP Right: {contingency['cp right']['reasonable']} reasonable, "
                f"{contingency['cp right']['unreasonable']} unreasonable\n"
                f"\tCP Wrong: {contingency['cp wrong']['reasonable']} reasonable, "
                f"{contingency['cp wrong']['unreasonable']} unreasonable\n"
            )


if __name__ == "__main__":
    args = parse_args()
    globals()[args.task](args)
