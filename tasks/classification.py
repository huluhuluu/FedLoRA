import torch
from transformers import AutoModelForSequenceClassification
from tasks import register_task
from tasks.base import BaseTask

TASK_FIELDS = {
    "agnews": ("text",),
    "banking77": ("text",),
    "sst2": ("sentence",),
    "cola": ("sentence",),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@register_task("classification")
class Task(BaseTask):
    model_class = AutoModelForSequenceClassification
    peft_task_type = "SEQ_CLS"
    label_kind = "categorical"
    metric_names = ("loss", "accuracy")

    def __init__(self, data_config, label_info):
        self.data_config, self.label_info = data_config, label_info

    def input_fields(self):
        return self.data_config.get(
            "text_fields", TASK_FIELDS.get(self.data_config.get("task"), ("text",))
        )

    def model_kwargs(self):
        return {
            "num_labels": self.label_info["num_labels"],
            "id2label": dict(enumerate(self.label_info["label_names"])),
            "label2id": {v: i for i, v in enumerate(self.label_info["label_names"])},
            "problem_type": "single_label_classification",
        }

    def head_modules(self, model):
        return [
            name
            for name in ("classifier", "score", "pre_classifier", "classification_head")
            if hasattr(model, name)
        ]

    def collate(self, tokenizer, rows):
        fields = self.input_fields()
        texts = [[row[field] for row in rows] for field in fields]
        batch = tokenizer(
            *texts,
            truncation=True,
            padding="max_length",
            max_length=self.data_config["seq_len"],
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(
            [self.label_info["label_mapping"][str(row["label"])] for row in rows]
        )
        return batch

    def statistics(self, output, batch):
        count = len(batch["labels"])
        return {
            "loss_sum": float(output.loss.detach()) * count,
            "weight": count,
            "correct": int(
                (output.logits.detach().argmax(-1) == batch["labels"]).sum()
            ),
        }

    def finalize(self, totals):
        return {
            "loss": totals["loss_sum"] / totals["weight"],
            "accuracy": totals.get("correct", 0) / totals["weight"],
        }
