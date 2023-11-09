from ds.hfdataset import HFDataset


class Arxiv(HFDataset):
    ds_name = "arxiv"
    dataset_kwargs = {
        "ds_name": "ccdv/arxiv-summarization",
        "ds_subset": "document",
        "col_map": {"article": "text", "abstract": "summary"},
    }

    def __init__(
        self,
        preview: bool,
        samples: int,
        min_input_size: int,
    ) -> None:
        super().__init__(
            preview=preview,
            samples=samples,
            min_input_size=min_input_size,
            **self.dataset_kwargs,
        )