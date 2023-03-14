from utils.utils_prompt import construct_prompt
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

COMPLETIONS_API_PARAMS_TURBO = {
    "temperature": 0.0,
    "max_tokens": 1000,
    "model": "gpt-3.5-turbo",
}

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = True,
    embedding_type: str = "hf",
    model_lang: str = "en",
    is_turbo: str = "false",
    messages: list = None,
    is_first_time: bool = True
) -> str:
    
    if is_first_time==True:

        prompt = construct_prompt(
        query,
        document_embeddings,
        df,
        embedding_type=embedding_type,
        model_lang=model_lang
    )
        if is_turbo=="true":
            messages.append({"role": "user", "content": prompt})
        
    else:
        prompt = query
        if is_turbo=="true":
            messages.append({"role": "user", "content": prompt})
    
    if show_prompt:
        print(prompt)

        
    if is_turbo!="true":
        response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    else:
        response = openai.ChatCompletion.create(
                messages=messages,
                **COMPLETIONS_API_PARAMS_TURBO,
            )
        

    if is_turbo=="true":
        messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"].strip(" \n")})
        return response["choices"][0]["message"]["content"].strip(" \n"), prompt, messages
    else:
        return response["choices"][0]["text"].strip(" \n"), prompt, messages