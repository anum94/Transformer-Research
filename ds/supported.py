import logging
from typing import Union

from ds.hfdataset import HFDataset

from ds.arxiv import Arxiv
from ds.govreport import GovReport
from ds.bigpatent import BigPatent
from ds.pubmed import Pubmed

translate_dataset_name = {
    "arxiv": Arxiv,
    "govreport": GovReport,
    "bigpatent": BigPatent,
    "pubmed": Pubmed,
}


def load_dataset(
    dataset: str,
    preview: bool = False,
    samples: Union[int, str] = "max",
    min_input_size: int = 0,
) -> HFDataset:
    logging.info(f"Preparing dataset {dataset}")

    assert dataset in translate_dataset_name
    return translate_dataset_name[dataset](
        preview=preview, samples=samples, min_input_size=min_input_size
    )
