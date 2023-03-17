from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
import time
import openai

EMBEDDING_MODEL = "text-embedding-ada-002"

model_language_map = {
    "en": "sentence-transformers/all-MiniLM-L6-v2",
    "tr": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
}


def get_hf_embeddings(text: str, model_lang='en') -> np.ndarray:
    """
    Returns the embeddings for the supplied text using the specified model. Uses the Hugging Face library.
    :param text: The text to embed.
    :param model_lang: The language of the model. Can be "en" or "tr".
    :return: The embeddings for the text.

    """

    model_name = model_language_map[model_lang]

    model = SentenceTransformer(model_name)
    sentence_embeddings = model.encode(text)
    sentence_embeddings = sentence_embeddings.reshape(1, -1)
    sentence_embeddings = normalize(sentence_embeddings)
    return sentence_embeddings[0]


def compute_doc_embeddings_hf(df: pd.DataFrame, model_lang='en') -> dict[tuple[str, str], np.ndarray]:
    """
    Computes the embeddings for the document sections.
    :param df: The dataframe containing the document sections.
    :param model_lang: The language of the model. Can be "en" or "tr".
    :return: The embeddings for the document sections.

    """

    return {
        idx: get_hf_embeddings(r.content, model_lang) for idx, r in df.iterrows()
    }


# below sourced from here https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    Returns the embeddings for the supplied text using the specified model. Uses the OpenAI API.
    :param text: The text to embed.
    :param model: The model to use.
    :return: The embeddings for the text.
    """
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    time.sleep(5)
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Computes the embeddings for the document sections.
    :param df: The dataframe containing the document sections.
    :return: The embeddings for the document sections.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }
