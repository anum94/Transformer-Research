from metrics.rouge import Rouge
from metrics.bertscore import BertScore

translate_metric_name = {"rouge": Rouge, "bertscore": BertScore}


def load_metrics(
    metrics: list,
) -> list:

    res = []
    for metric in metrics:
        assert metric in translate_metric_name
        res.append(translate_metric_name[metric]())

    return res
