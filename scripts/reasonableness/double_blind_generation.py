import os
import yaml
import random
import csv

import gridparse


def parse_args():
    parser = gridparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--experiment_dir", nargs="+", required=True)
    parser.add_argument("--add_experiment_dir", nargs="*")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output-dir", required=True)

    return parser.parse_args()


def icl_vs_gt(args):
    for experiment_dir in args.experiment_dir:
        with open(os.path.join(experiment_dir, "indexed_metrics.yml")) as fp:
            experiments = yaml.safe_load(fp)

        wrong = {}

        for exp_no, indexed_metrics in experiments.items():
            for example_id, measurements in indexed_metrics.items():
                if example_id == "description":
                    continue
                if "test_gt" in measurements:
                    gt = measurements["test_gt"]
                    pred = measurements["test_preds"]
                    out = measurements["test_outs"]
                else:
                    gt = measurements["test_true"]
                    pred = measurements["test_pred"]
                    out = measurements["test_out"]
                if out.startswith("{") and "}" not in out:
                    continue

                if set(gt) != set(pred):
                    _id = example_id + "-" + exp_no
                    wrong[_id] = dict(
                        pred=pred, gt=gt, text=measurements["test_text"]
                    )

        keys = random.sample(list(wrong), args.n)

        experiment_dir, experiment_subfolder = os.path.split(
            os.path.abspath(experiment_dir)
        )
        experiment_folder = os.path.split(experiment_dir)[1]
        analysis_dir = os.path.join(
            args.output_dir, experiment_folder, experiment_subfolder
        )
        os.makedirs(analysis_dir, exist_ok=True)

        first_labels = [random.randint(0, 1) for _ in keys]

        with open(os.path.join(analysis_dir, "more_correct.csv"), "w") as fp:
            writer = csv.writer(fp)
            writer.writerow(["text", "labels_1", "labels_2", "best"])
            for key, which in zip(keys, first_labels):
                row = wrong[key]
                if which == 0:
                    writer.writerow([row["text"], row["pred"], row["gt"]])
                else:
                    writer.writerow([row["text"], row["gt"], row["pred"]])

        with open(
            os.path.join(analysis_dir, "more_correct-labels.csv"), "w"
        ) as fp:
            writer = csv.writer(fp)
            writer.writerow(["id", "labels"])
            for key, which in zip(keys, first_labels):
                writer.writerow([key, which])


def query_prompt_vs_gt(args):
    for experiment_dir in args.experiment_dir:
        with open(os.path.join(experiment_dir, "indexed_metrics.yml")) as fp:
            experiments = yaml.safe_load(fp)

        correct = {}
        wrong = {}
        example_ids = set()

        for i, (exp_no, indexed_metrics) in enumerate(experiments.items()):
            for example_id, measurements in indexed_metrics.items():
                if example_id == "description":
                    continue
                if "test_gt" in measurements:
                    gt = measurements["test_gt"]
                    pred = measurements["test_preds"]
                    out = measurements["test_outs"]
                else:
                    gt = measurements["test_true"]
                    pred = measurements["test_pred"]
                    out = measurements["test_out"]
                if out.startswith("{") and "}" not in out:
                    continue

                _id = example_id + "-" + exp_no

                if set(gt) == set(pred) and i == 0:
                    correct[_id] = dict(gt=gt, text=measurements["test_text"])
                elif set(gt) != set(pred) and example_id not in example_ids:
                    wrong[_id] = dict(
                        pred=pred, gt=gt, text=measurements["test_text"]
                    )
                    example_ids.add(example_id)

        pred_keys = random.sample(list(wrong), min(2 * args.n, len(wrong)))
        first_labels = [random.randint(0, 1) for _ in pred_keys]

        experiment_dir, experiment_subfolder = os.path.split(
            os.path.abspath(experiment_dir)
        )
        experiment_folder = os.path.split(experiment_dir)[1]
        analysis_dir = os.path.join(
            args.output_dir, experiment_folder, experiment_subfolder
        )
        os.makedirs(analysis_dir, exist_ok=True)

        with open(os.path.join(analysis_dir, "more_correct.csv"), "w") as fp:
            writer = csv.writer(fp)
            writer.writerow(["text", "labels_1", "labels_2", "best"])
            for key, which in zip(pred_keys, first_labels):
                row = wrong[key]
                if which == 0:
                    writer.writerow([row["text"], row["pred"], row["gt"]])
                else:
                    writer.writerow([row["text"], row["gt"], row["pred"]])

        with open(
            os.path.join(analysis_dir, "more_correct-labels.csv"), "w"
        ) as fp:
            writer = csv.writer(fp)
            writer.writerow(["id", "labels"])
            for key, which in zip(pred_keys, first_labels):
                writer.writerow([key, which])

        if args.n > len(wrong):
            print(f"Only {len(wrong)} errors, not {args.n}")

        pred_keys = random.sample(list(wrong), min(args.n, len(wrong)))
        gt_keys = random.sample(list(correct), args.n)

        queue = [0] * min(args.n, len(wrong)) + [1] * args.n
        random.shuffle(queue)
        final_keys = []

        with open(os.path.join(analysis_dir, "errors.csv"), "w") as fp:
            writer = csv.writer(fp)
            writer.writerow(["text", "labels", "reasonable"])
            for which in queue:
                if which == 0:
                    key = pred_keys.pop()
                    row = wrong[key]
                else:
                    key = gt_keys.pop()
                    row = correct[key]

                writer.writerow([row["text"], row["gt"]])
                final_keys.append(key)

        with open(os.path.join(analysis_dir, "errors-labels.csv"), "w") as fp:
            writer = csv.writer(fp)
            writer.writerow(["id", "labels"])
            for key, which in zip(final_keys, queue):
                writer.writerow([key, which])
            # fp.write("\n".join(str(x) for x in queue))


def baseline_vs_gt(args):
    if not args.add_experiment_dir:
        args.add_experiment_dir = [None] * len(args.experiment_dir)
    for experiment_dir, add_experiment_dir in zip(
        args.experiment_dir, args.add_experiment_dir
    ):
        with open(os.path.join(experiment_dir, "indexed_metrics.yml")) as fp:
            experiments = yaml.safe_load(fp)

        if add_experiment_dir:
            with open(
                os.path.join(add_experiment_dir, "indexed_metrics.yml")
            ) as fp:
                experiments_for_label = yaml.safe_load(fp)
        else:
            experiments_for_label = experiments

        reasonable = {}
        unreasonable = {}
        example_ids = set()

        for i, (exp_no, indexed_metrics) in enumerate(experiments.items()):
            for example_id, measurements in indexed_metrics.items():
                if example_id == "description":
                    continue
                if add_experiment_dir:
                    if "test_gt" in experiments_for_label[exp_no][example_id]:
                        gt = experiments_for_label[exp_no][example_id][
                            "test_gt"
                        ]
                    else:
                        gt = experiments_for_label[exp_no][example_id][
                            "test_true"
                        ]
                else:
                    gt = measurements["test_checked_label"]
                pred = measurements["test_preds"]
                out = measurements["test_outs"]
                if out.startswith("{") and "}" not in out:
                    continue

                _id = example_id + "-" + exp_no

                if pred in ("reasonable", "yes") and i == 0:
                    reasonable[_id] = dict(
                        gt=gt, text=measurements["test_text"]
                    )
                elif (
                    pred in ("unreasonable", "no")
                    and example_id not in example_ids
                ):
                    unreasonable[_id] = dict(
                        pred=pred, gt=gt, text=measurements["test_text"]
                    )
                    example_ids.add(example_id)

        experiment_dir, experiment_subfolder = os.path.split(
            os.path.abspath(experiment_dir)
        )
        experiment_folder = os.path.split(experiment_dir)[1]
        analysis_dir = os.path.join(
            args.output_dir, experiment_folder, experiment_subfolder
        )
        os.makedirs(analysis_dir, exist_ok=True)

        if args.n > len(unreasonable):
            print(f"Only {len(unreasonable)} errors, not {args.n}")

        pred_keys = random.sample(
            list(unreasonable), min(args.n, len(unreasonable))
        )
        gt_keys = random.sample(list(reasonable), args.n)

        queue = [0] * min(args.n, len(unreasonable)) + [1] * args.n
        random.shuffle(queue)
        final_keys = []

        with open(os.path.join(analysis_dir, "errors.csv"), "w") as fp:
            writer = csv.writer(fp)
            writer.writerow(["text", "labels", "reasonable"])
            for which in queue:
                if which == 0:
                    key = pred_keys.pop()
                    row = unreasonable[key]
                else:
                    key = gt_keys.pop()
                    row = reasonable[key]

                writer.writerow([row["text"], row["gt"]])
                final_keys.append(key)

        with open(os.path.join(analysis_dir, "errors-labels.csv"), "w") as fp:
            writer = csv.writer(fp)
            writer.writerow(["id", "labels"])
            for key, which in zip(final_keys, queue):
                writer.writerow([key, which])
            # fp.write("\n".join(str(x) for x in queue))


if __name__ == "__main__":
    args = parse_args()
    globals()[args.task](args)
