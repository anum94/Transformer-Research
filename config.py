import argparse
import time
import os
from json import load
from typing import Sequence

from dotenv import load_dotenv
from huggingface_hub.hf_api import ModelInfo
import torch

from utils.hfrepo import get_available_models

import wandb


class Config:
    tokenizer_parallelism_key: str = "TOKENIZERS_PARALLELISM"
    hf_token_key: str = "HF_API_TOKEN"
    hf_user_key: str = "HF_USER"
    hf_hub_cache_key: str = "TRANSFORMERS_CACHE"
    hf_ds_cache_key: str = "HF_DATASETS_CACHE"
    container_prefix: str = "/mnt/container/"
    ckpt_dir_path: str = os.path.join(container_prefix, "ckpt")
    hf_hub_cache_path: str = os.path.join(container_prefix, ".cache/huggingface/hub")
    hf_ds_cache_path: str = os.path.join(
        container_prefix, ".cache/huggingface/datasets"
    )

    wandb_token_key: str = "WANDB_TOKEN"
    wandb_log_key: str = "WANDB_LOG_MODEL"
    wandb_watch_key: str = "WANDB_WATCH"

    exec_args: dict = {}
    exec_kwargs: dict = {}
    exec_string: str = ""
    exec_timestamp: str = ""

    method: str = ""
    dsconfig: str = ""
    working_dir: str = ""
    ckpt: str = ""

    hf_token: str = ""
    hf_user: str = ""
    available_models: Sequence[ModelInfo] = []

    def __init__(self) -> None:
        # TODO validate json config
        # TODO extend console config

        self.parse_args(self.configure_parser())
        self.configure_env()

        self.exec_string = f"{self.exec_args['model']}-{self.exec_args['dataset']}-{self.exec_kwargs['enc_max_len']}"
        self.exec_timestamp = time.strftime("%d%m%Y%H%M%S", time.localtime())
        self.working_dir = os.path.dirname(os.path.abspath(__file__))

        self.available_models = None#get_available_models(self.hf_user)

    def configure_parser(self) -> dict:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, help="configuration file path")
        parser.add_argument("--dsconfig", type=str, help="deepspeed config file")
        parser.add_argument("--method", type=str, help="program execution mode")
        parser.add_argument("--model", type=str, help="model name")
        parser.add_argument("--dataset", type=str, help="data name")
        parser.add_argument("--metrics", nargs="+", help="list metrics to evaluate")
        parser.add_argument(
            "--preview", type=bool, default=False, help="preview of execution"
        )
        parser.add_argument("--ckpt", type=str, help="resume from checkpoint")
        return vars(parser.parse_args())

    def parse_args(self, args: dict) -> None:
        if args["config"]:
            with open(args["config"], "r") as fp:
                json = load(fp)
                self.method = json["method"]
                self.exec_args = json["exec_args"]
                self.exec_kwargs = json["exec_kwargs"]

        self.ckpt = args["ckpt"] if args["ckpt"] else None
        self.dsconfig = args["dsconfig"] if args["dsconfig"] else None

    def configure_env(self) -> str:
        load_dotenv()

        # distributed setup
        if torch.cuda.is_available():
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
        os.environ[self.tokenizer_parallelism_key] = "true"

        # hf setup
        token = os.environ.get(self.hf_token_key)
        assert token and token != "<token>", "HuggingFace API token is not defined"
        user = os.environ.get(self.hf_user_key)
        assert user and user != "<user>", "HuggingFace user is not defined"
        self.hf_token, self.hf_user = token, user

        os.environ[self.hf_ds_cache_key] = self.hf_ds_cache_path
        os.environ[self.hf_hub_cache_key] = self.hf_hub_cache_path

        # wandb setup
        wandb_tok = os.environ.get(self.wandb_token_key)
        assert wandb_tok and wandb_tok != "<wb_token>", "Wandb token is not defined"
        wandb.login(anonymous="allow", key=wandb_tok)

        # os.environ["WANDB_PROJECT"]="my-awesome-project"
        os.environ[self.wandb_log_key] = "false"
        os.environ[self.wandb_watch_key] = "false"

    def ssh_key_path(self) -> str:
        return f"{self.working_dir}/.ssh/id_ed25519"

    def log_path(self) -> str:
        return f"{self.working_dir}/results/{self.exec_string}-{self.exec_timestamp}"

    def ckpt_log_path(self) -> str:
        return f"{self.ckpt_dir_path}/{self.exec_string}-{self.exec_timestamp}"

    def prev_log_path(self) -> str:
        # remove the last /checkpoint-1600 from:
        # /dss/dsshome1/0F/ge58hep2/transformer-research/results/longt5-pubmed-4096-10052023004612/checkpoint-1600
        return self.ckpt.rsplit("/", 1)[0]


config = Config()
