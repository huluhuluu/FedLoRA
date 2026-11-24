import math
from transformers import AutoModelForCausalLM
from tasks import register_task
from tasks.base import BaseTask


@register_task("causal_lm")
class Task(BaseTask):
    model_class = AutoModelForCausalLM
    peft_task_type = "CAUSAL_LM"
    metric_names = ("loss", "perplexity")

    def __init__(self, data_config, label_info):
        self.data_config = data_config

    def input_fields(self):
        return self.data_config.get("text_fields", ["text"])

    def collate(self, tokenizer, rows):
        field = self.input_fields()[0]
        batch = tokenizer(
            [row[field] for row in rows],
            truncation=True,
            padding="max_length",
            max_length=self.data_config["seq_len"],
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        if not (labels[:, 1:] != -100).any():
            raise ValueError("Causal LM batches need at least two unmasked tokens")
        batch["labels"] = labels
        return batch

    def statistics(self, output, batch):
        count = int((batch["labels"][:, 1:] != -100).sum())
        return {"loss_sum": float(output.loss.detach()) * count, "weight": count}

    def finalize(self, totals):
        loss = totals["loss_sum"] / totals["weight"]
        return {
            "loss": loss,
            "perplexity": math.exp(loss) if loss < 700 else None,
        }
