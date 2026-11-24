"""Default loss and validation for text tasks."""


class BaseTask:
    label_kind = None
    metric_names = ("loss",)

    def model_kwargs(self):
        return {}

    def head_modules(self, model):
        return None

    def compute_loss(self, output, batch):
        return output.loss

    def validate_data(self, bundle):
        required = set(self.input_fields()) | ({"label"} if self.label_kind else set())
        for dataset in (bundle.train, bundle.evaluation):
            if dataset is None:
                continue
            missing = required - set(dataset.column_names)
            if missing:
                raise ValueError(f"Task data missing columns: {sorted(missing)}")
