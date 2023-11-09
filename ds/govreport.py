from ds.hfdataset import HFDataset


class GovReport(HFDataset):
    ds_name = "govreport"
    dataset_kwargs = {
        "ds_name": "ccdv/govreport-summarization",
        "ds_subset": "document",
        "col_map": {"report": "text"},
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
