from typing import List
import os
from ..utils.utils_docs import extract_paragraphs
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context
from io import BytesIO


def docs_scrape(mongo_client, file_path: str, query: str,max_tokens, embedding_extractor: str = "hf", model_lang: str = "en", to_save: str = "false", is_turbo: str = "false", messages: List[dict] = []) -> dict:
    """
    Extracts paragraphs from a Word file and computes embeddings for each paragraph, then answers a query using the embeddings.
    :param file_path: Path to the Word file
    :param query: Query to answer
    :param max_tokens: Maximum number of tokens to generate
    :param embedding_extractor: Extractor to use for computing embeddings. Options are "hf" and "openai"
    :param model_lang: Language of the model to use for computing embeddings. Options are "en" and "tr"
    :param to_save: Whether to save the embeddings to MongoDB
    :param is_turbo: Whether to use turbo mode
    :param messages: Previous messages
    :return: Answer to the query and the prompt and messages
    """
    print("Processing Word file...")

    if not os.path.isfile(file_path):
        return {"error": "File not found"}

    _, ext = os.path.splitext(file_path)
    allowed_ext = [".doc", ".docx"]
    if ext not in allowed_ext:
        return {"error": "Only Word files are allowed"}
        

    docs_df = None
    docs_embeddings = None

    print("Extracting paragraphs...")

    with open(file_path, "rb") as f:
        docs_buffer = BytesIO(f.read())

    docs_df = extract_paragraphs(docs_buffer)

    print("Computing embeddings...")

    if docs_embeddings is None:
        if embedding_extractor == "hf":
           docs_embeddings = compute_doc_embeddings_hf(docs_df, model_lang)
        else:
           docs_embeddings = compute_doc_embeddings(docs_df)

    target = query
    answer = ""

    print("Answering query...")

    if len(messages) == 0 and is_turbo == "true":
        messages = [{"role": "system", "content": "you are a helpful assistant"}]

    is_first_time = True
    if len(messages) > 2:
        is_first_time = False
        print("not first time")

    if embedding_extractor == "hf":
        answer, prompt, messages = answer_query_with_context(target, docs_df, docs_embeddings, embedding_type="hf", model_lang=model_lang, is_turbo=is_turbo, messages=messages, is_first_time=is_first_time, max_tokens=max_tokens)
    else:
        answer, prompt, messages = answer_query_with_context(target, docs_df, docs_embeddings, embedding_type="openai", model_lang=model_lang, is_turbo=is_turbo, messages=messages, is_first_time=is_first_time, max_tokens=max_tokens)

    if to_save == "true" and is_turbo == "false":
        mongo_client.pairs_docs.insert_one({"query": target, "answer": answer, "prompt": prompt})
    else:
        mongo_client.pairs_docs_turbo.insert_one({"conversation": messages})

    print("Done!")

    return answer, prompt, messages
