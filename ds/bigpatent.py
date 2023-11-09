from ds.hfdataset import HFDataset


class BigPatent(HFDataset):
    ds_name = "bigpatent"
    dataset_kwargs = {
        "ds_name": "big_patent",
        "ds_subset": "all",
        "col_map": {"description": "text", "abstract": "summary"},
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
            **self.dataset_kwargs
        )
