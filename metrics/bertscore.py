import evaluate
import numpy as np

from metrics.metric import Metric


class BertScore(Metric):
    metric_name = "bertscore"

    def __init__(self) -> None:
        evaluate.logging.set_verbosity_warning()
        self.metric = evaluate.load(self.metric_name)

    def compute(self, predictions: list, references: list) -> dict:
        # uses roberta-large as default
        res = self.metric.compute(
            predictions=predictions, references=references, lang="en"
        )

        for k, v in res.items():
            res[k] = np.mean(v)

        return res
