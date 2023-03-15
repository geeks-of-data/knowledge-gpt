from typing import Optional
from ..utils.utils_scrape import scrape_content
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context


def scrape_website(mongo_client, url: str,max_tokens, embedding_extractor: str, model_lang: str, query: Optional[str] = None, to_save: bool = False):
    """
    Function that takes a URL as input and returns the response answer.
    :param url: URL to scrape
    :param embedding_extractor: Extractor to use for computing embeddings. Options are "hf" and "openai"
    :param max_tokens: Maximum number of tokens to use for the prompt
    :param model_lang: Language of the model to use for computing embeddings. Options are "en" and "tr"
    :param query: Query to answer
    :param to_save: Whether to save the embeddings to MongoDB
    :return: Answer to the query and the prompt and messages
    
    """
    print("Scraping website...")
    if not url:
        raise ValueError("URL is missing")

    web_df = None
    web_embeddings = None

    web_df = scrape_content(url)

    print("Computing embeddings...")

    if web_embeddings is None:
        if embedding_extractor == "hf":
           web_embeddings = compute_doc_embeddings_hf(web_df, model_lang)
        else:
           web_embeddings = compute_doc_embeddings(web_df)

    target = query
    answer = ""

    print("Answering query...")

    if embedding_extractor == "hf":
        answer, prompt, messages = answer_query_with_context(target, web_df, web_embeddings, embedding_type="hf", model_lang=model_lang, max_tokens=max_tokens)
    else:
        answer, prompt, messages = answer_query_with_context(target, web_df, web_embeddings, embedding_type="openai", model_lang=model_lang, max_tokens=max_tokens)

    if to_save:
        mongo_client.pair_web.insert_one({"query": target, "answer": answer, "prompt": prompt})

    print("Done!")

    return answer, prompt, messages
