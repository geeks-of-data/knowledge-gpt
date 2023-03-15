from ..utils.utils_powerpoint import process_pptx
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context
from io import BytesIO
import os


def powerpoint_scrape(mongo_client, file_path: str,max_tokens, query: str = "", embedding_extractor: str = "hf", model_lang: str = "en", to_save: bool = False):
    """
    Function that takes a file path for a Powerpoint presentation as input and returns the response answer.

    :param file_path: Path to the Powerpoint file
    :param query: Query to answer
    :param max_tokens: Maximum number of tokens to use for the prompt
    :param embedding_extractor: Extractor to use for computing embeddings. Options are "hf" and "openai"
    :param model_lang: Language of the model to use for computing embeddings. Options are "en" and "tr"
    :param to_save: Whether to save the embeddings to MongoDB
    :return: Answer to the query and the prompt and messages
    
    """
    print("Processing Powerpoint file...")

    if not os.path.isfile(file_path):
        raise ValueError("Invalid file path provided.")

    if not file_path.endswith(".pptx"):
        raise ValueError("Only Powerpoint (.pptx) files are allowed.")


    pptx_df = None
    pptx_embeddings = None

    print("Extracting paragraphs...")

    with open(file_path, "rb") as f:
        pptx_buffer = BytesIO(f.read())

    pptx_df = process_pptx(pptx_buffer)

    print("Computing embeddings...")

    if pptx_embeddings is None:
        if embedding_extractor == "hf":
            pptx_embeddings = compute_doc_embeddings_hf(pptx_df, model_lang)
        else:
            pptx_embeddings = compute_doc_embeddings(pptx_df)

    target = query
    answer = ""

    print("Answering query...")

    if embedding_extractor == "hf":
        answer, prompt, messages = answer_query_with_context(target, pptx_df, pptx_embeddings, embedding_type="hf", model_lang=model_lang, max_tokens=max_tokens)
    else:
        answer, prompt, messages = answer_query_with_context(target, pptx_df, pptx_embeddings, embedding_type="openai", model_lang=model_lang, max_tokens=max_tokens)

    if to_save:
        mongo_client.pair_pptx.insert_one({"query": target, "answer": answer, "prompt": prompt})

    print("Done!")

    return answer, prompt, messages
