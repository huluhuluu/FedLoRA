"""Example task plugin: single-label classification with macro F1."""

import torch
from tasks import register_task
from tasks.classification import Task


@register_task("classification_f1")
class ClassificationF1(Task):
    metric_names = ("loss", "accuracy", "macro_f1")

    def statistics(self, output, batch):
        values = super().statistics(output, batch)
        classes = self.label_info["num_labels"]
        counts = (
            torch.bincount(
                batch["labels"] * classes + output.logits.detach().argmax(-1),
                minlength=classes * classes,
            )
            .cpu()
            .tolist()
        )
        values.update({f"cm_{i}": value for i, value in enumerate(counts) if value})
        return values

    def finalize(self, totals):
        values = super().finalize(totals)
        classes = self.label_info["num_labels"]
        scores = []
        for label in range(classes):
            tp = totals.get(f"cm_{label * classes + label}", 0)
            predicted = sum(
                totals.get(f"cm_{row * classes + label}", 0) for row in range(classes)
            )
            actual = sum(
                totals.get(f"cm_{label * classes + column}", 0)
                for column in range(classes)
            )
            scores.append(2 * tp / (predicted + actual) if predicted + actual else 0)
        values["macro_f1"] = sum(scores) / classes
        return values
