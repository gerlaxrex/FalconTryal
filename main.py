from app.services.falcon_model import get_model_pipeline
from transformers import Pipeline, PreTrainedTokenizer


def perform_query(query: str,
                  pipeline_model: Pipeline,
                  tokenizer_model: PreTrainedTokenizer):

    sequences = pipeline_model(
        query,
        max_length=50,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer_model.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


if __name__ == '__main__':
    MODEL_NAME = "gpt2"
    pipeline, tokenizer = get_model_pipeline(model_name=MODEL_NAME)
    perform_query("Hi, what's your name?", pipeline, tokenizer)
