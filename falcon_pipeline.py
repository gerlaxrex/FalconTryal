import functools
from typing import Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM, Pipeline, PreTrainedTokenizer
import transformers
import torch


@functools.lru_cache(maxsize=1)
def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


@functools.lru_cache(maxsize=1)
def get_model_pipeline(model_name: str,
                       tokenizer_model: PreTrainedTokenizer = None) -> Tuple[Pipeline, PreTrainedTokenizer]:
    tokenizer = tokenizer_model or get_tokenizer(model_name)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    return pipeline, tokenizer
