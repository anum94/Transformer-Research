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

        # conditions for admitting data into the training:
        # 1) Text (x) is twice as long as summary (y) and less than 1000 times longer.
        # 2) Summary is not a verbatim part of the text.
        # 3) The text document has a minimum length (min_input_size).
        def mask(x, y):
            return (
                    2 * len(y) < len(x) < 1000 * len(y)
                    and y not in x
                    and len(x) >= min_input_size
            )

        def fn(batch: dict):
            res = {"text": [], "summary": []}
            z = zip(batch["text"], batch["summary"])
            # apply the logical inverse of `mask` to obtain admissible documents.
            valid = list(filter(lambda x: not mask(x[0], x[1]), z))
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
