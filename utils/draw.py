"""Plot training or versioned evaluation metrics inside an experiment."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--metric", default="eval_loss")
    args = parser.parse_args()
    if not args.metric.replace("_", "").isalnum():
        raise ValueError("Metric must be a CSV column name")
    from utils.helper import check_output_path, configure_runtime

    configure_runtime(2)

    output = check_output_path(args.run.resolve() / (args.metric + ".png"))
    source = (
        args.run / "evaluations.csv"
        if args.metric.startswith("eval_") and (args.run / "evaluations.csv").exists()
        else args.run / "rounds.csv"
    )
    with source.open() as stream:
        reader = csv.DictReader(stream)
        if args.metric not in reader.fieldnames:
            raise ValueError(f"Unknown metric; available: {reader.fieldnames}")
        rows = [row for row in reader if row.get(args.metric)]
    if not rows:
        raise ValueError("No observations for this metric")
    xkey = "index" if "index" in rows[0] else "step" if rows[0].get("step") else "round"
    rows.sort(key=lambda row: int(row[xkey]))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    axes.plot(
        [int(row[xkey]) for row in rows],
        [float(row[args.metric]) for row in rows],
        marker="o",
    )
    axes.set(xlabel=rows[0].get("unit", xkey), ylabel=args.metric)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


if __name__ == "__main__":
    main()
