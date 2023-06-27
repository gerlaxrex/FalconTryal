import logging
from typing import Tuple, Annotated

import cachetools
from fastapi import APIRouter, Depends
from transformers import Pipeline, PreTrainedTokenizer

from app.services.falcon_model import get_model_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL_NAME = "tiiuae/falcon-7b"


class PretrainedTokenizer:
    pass


@cachetools.cached(cache=cachetools.LRUCache(maxsize=1))
def get_pipeline(model_name: str = MODEL_NAME) -> Tuple[Pipeline, PreTrainedTokenizer]:
    return get_model_pipeline(model_name)


@router.post(path="/generate")
def generate_model_response(query: str,
                            pipeline: Annotated[Pipeline, Depends(get_model)]):
    pipeline[0](query,
                max_length=50,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=pipeline[1].eos_token_id
                )
