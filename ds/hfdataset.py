import logging
from typing import Union
from datasets import load_dataset, DatasetDict, Dataset


class HFDataset:
    ds: DatasetDict
    ds_name: str

    def __init__(
        self,
        ds_name: str,
        ds_subset: str,
        col_map: dict,
        preview: bool,
        preview_size: int = 4,
        min_input_size: int = 0,
        samples: Union[int, str] = "max",
    ) -> None:
        data = load_dataset(ds_name, ds_subset)

        if preview:
            for k in data.keys():
                data[k] = data[k].select(range(preview_size))

        elif samples != "max":
            data["train"] = data["train"].select(
                range(min(len(data["train"]), samples))
            )

        self.ds = self.preprocess(data, col_map, min_input_size)

    def get_split(self, key: str) -> Dataset:
        return self.ds[key]

    def scrolls_preprocess(
        self,
        data: DatasetDict,
        col_map: dict,
        min_input_size: int,
    ) -> DatasetDict:
        # removing samples where text is shorter than 2x summary
        # where text is bigger than 1000x summary
        # where summary is verbatim in text
        # as per SCROLLS

        # samples smaller than min_input_size also removed (defaults to 0)
        # test ds remains the same

        def mask(x, y):
            return (
                len(x) > 2 * len(y)
                and len(x) < 1000 * len(y)
                and y not in x
                and len(x) >= min_input_size
            )

        def fn(batch: dict):
            res = {"text": [], "summary": []}
            z = zip(batch["text"], batch["summary"])
            valid = list(filter(lambda x: mask(x[0], x[1]), z))
            res["text"] = [valid[idx][0] for idx in range(len(valid))]
            res["summary"] = [valid[idx][1] for idx in range(len(valid))]
            return res

        logging.info("Preprocessing dataset")
        data = data.rename_columns(col_map)
        save_test = data["test"]
        data = data.map(fn, batched=True)
        data["test"] = save_test
        data.set_format("torch")
        return data

    def preprocess(
        self,
        data: DatasetDict,
        col_map: dict,
        min_input_size: int,
    ) -> DatasetDict:
        # subclasses can implement custom behaviour by defining the preprocess fn
        return self.scrolls_preprocess(
            data=data, col_map=col_map, min_input_size=min_input_size
        )
