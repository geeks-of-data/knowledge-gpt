# https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb sourced from here
from knowledgegpt.utils.utils_prompt import construct_prompt
import openai
import pandas as pd
import numpy as np
import tiktoken

COMPLETIONS_API_PARAMS = {
    "temperature": 0.0,
    "model": "text-davinci-003",
    "max_tokens": 1000,
}

COMPLETIONS_API_PARAMS_TURBO = {
    "temperature": 0.0,
    "model": "gpt-3.5-turbo",
    "max_tokens": 1000,
}


def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        document_embeddings: dict[(str, str), np.array],
        verbose: bool = False,
        embedding_type: str = "hf",
        model_lang: str = "en",
        is_turbo: str = False,
        messages: list = None,
        index_type: str = "basic",
        max_tokens=1000
) -> str:
    """
    Answer a query using the provided context.
    :param query: The query to answer.
    :param df: The dataframe containing the document sections.
    :param document_embeddings: The embeddings of the document sections.    
    :param show_prompt: Whether to print the prompt.
    :param embedding_type: The type of embedding used. Can be "hf" or "tf".
    :param model_lang: The language of the model. Can be "en" or "tr".
    :param is_turbo: Whether to use turbo model or not. Can be "true" or "false".
    :param messages: The messages to be used in turbo model.
    :param is_first_time: Whether it is the first time to use turbo model or not.
    :param max_tokens: The maximum number of tokens to be used in turbo model.
    :return: The answer to the query.

    """
    
    if len(messages) < 3 or not is_turbo:
        prompt = construct_prompt(
            verbose=verbose,
            question=query,
            context_embeddings=document_embeddings,
            df=df,
            embedding_type=embedding_type,
            model_lang=model_lang,
            max_tokens=max_tokens,
            index_type=index_type
        )
        if is_turbo:
            messages.append({"role": "user", "content": prompt})
    else:
        prompt = query
        if is_turbo:
            messages.append({"role": "user", "content": prompt})

    encoding = encoding = tiktoken.get_encoding("gpt2")
    if is_turbo:
        messages_token_length = encoding.encode(str(messages))
        if len(messages_token_length) > 4096:
            
            del messages[2:4]


    
    if not verbose:
        print(prompt)


    if not is_turbo:
        prompt_len = len(encoding.encode(prompt))
        COMPLETIONS_API_PARAMS["max_tokens"] =  2000 - prompt_len

        response = openai.Completion.create(
            prompt=prompt,
            **COMPLETIONS_API_PARAMS
        )
    else:

        messages_token_length = encoding.encode(str(messages))
        COMPLETIONS_API_PARAMS_TURBO["max_tokens"] = 4096 - len(messages_token_length)

        response = openai.ChatCompletion.create(

            messages=messages,
            **COMPLETIONS_API_PARAMS_TURBO,
        )

    if is_turbo:
        messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"].strip(" \n")})
        return response["choices"][0]["message"]["content"].strip(" \n"), prompt, messages
    else:
        return response["choices"][0]["text"].strip(" \n"), prompt, messages
