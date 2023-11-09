from typing import Sequence

from exec.eval import Evaluation
from exec.finetune import Finetune
from exec.execute import Execute

from models.model import Model
from ds.hfdataset import HFDataset
from metrics.metric import Metric
from config import Config

translate_exec_name = {
    "eval": Evaluation,
    "finetune": Finetune,
}


def load_execute(
    method: str,
    model: Model,
    dataset: HFDataset,
    metrics: Sequence[Metric],
    config: Config,
    kwargs: dict,
) -> None:

    assert method in translate_exec_name, "Unrecognized execution method"
    exec_method: Execute = translate_exec_name[method](
        model=model,
        dataset=dataset,
        metrics=metrics,
        config=config,
        kwargs=kwargs,
    )

    exec_method.execute()
