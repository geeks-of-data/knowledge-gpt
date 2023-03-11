from utils_prompt import construct_prompt
import openai
import pandas as pd
import numpy as np
from config import SECRET_KEY

openai.api_key = SECRET_KEY

COMPLETIONS_API_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 1000,
    "model": "text-davinci-003",
}

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = True
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")