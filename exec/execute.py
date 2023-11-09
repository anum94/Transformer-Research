from typing import Sequence

from metrics.metric import Metric
from models.hfmodel import HFModel
from ds.hfdataset import HFDataset
from config import Config


class Execute:
    def __init__(
        self,
        model: HFModel,
        dataset: HFDataset,
        metrics: Sequence[Metric],
        config: Config,
        kwargs: dict,
    ) -> None:
        pass

    def execute(self):
        pass
