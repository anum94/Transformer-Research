import logging
from typing import Sequence
from huggingface_hub import login, ModelFilter, hf_api
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments


def hf_login(token: str) -> None:
    login(token)


def get_model_id(user: str, path: str) -> str:
    return f"{user}/{path}"


def model_is_available(available_models: Sequence[hf_api.ModelInfo], path: str) -> bool:
    return any(m.modelId == path for m in available_models)


def get_epoch_from_repo_commit(path: str) -> int:
    # TODO ideally we get the commit message from the hf repo URL and dynamically configure the epoch nr
    # https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/trainer.py#L3442
    # commit_message = f"Training in progress, epoch {int(self.state.epoch)}
    pass


def get_available_models(user: str) -> list:
    filter = ModelFilter(author=user)
    return [model for model in hf_api.list_models(filter=filter)]


class SaveCallback(TrainerCallback):
    def __init__(self, token, hf_path) -> None:
        self.hf_token = token
        self.hf_path = hf_path

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs,
    ):
        try:
            logging.info(f"pushing tokenizer to {self.hf_path}")
            tokenizer.push_to_hub(
                repo_id=self.hf_path,
                use_auth_token=self.hf_token,
            )

        except Exception as e:
            logging.error(f"Error pushing tokenizer to HF\n{e}")

        try:
            logging.info(f"pushing model to {self.hf_path}")
            ckpt_dir = get_last_checkpoint(args.output_dir)
            model = load_state_dict_from_zero_checkpoint(
                model=model, checkpoint_dir=ckpt_dir
            )
            model.push_to_hub(
                repo_id=self.hf_path,
                commit_message=f"Model saved at the end of {args.num_train_epochs} epochs",
                use_auth_token=self.hf_token,
            )

        except Exception as e:
            logging.error(f"Error pushing model to HF\n{e}")
