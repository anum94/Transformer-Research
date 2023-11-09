import logging
from config import config

from ds.supported import load_dataset
from models.supported import load_model
from metrics.supported import load_metrics
from exec.supported import load_execute

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    assert "model" in config.exec_args, "Please provide a model name"
    assert all(
        [
            key in config.exec_args
            for key in ("dataset", "preview", "samples", "min_input_size")
        ]
    ), "Please provide accurate dataset config"
    assert "metrics" in config.exec_args, "Please provide a metrics list"

    method = config.method
    model = load_model(config.exec_args.get("model"))
    metrics = load_metrics(config.exec_args.get("metrics"))
    dataset = load_dataset(
        dataset=config.exec_args.get("dataset"),
        preview=config.exec_args.get("preview"),
        samples=config.exec_args.get("samples"),
        min_input_size=config.exec_args.get("min_input_size"),
    )

    execute = load_execute(
        method=config.method,
        model=model,
        metrics=metrics,
        dataset=dataset,
        config=config,
        kwargs=config.exec_kwargs,
    )
