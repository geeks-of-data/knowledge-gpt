from ..utils.utils_subtitles import scrape_youtube
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context

def yt_subs_scrape(mongo_client, video_id: str, query: str, max_tokens, model_lang: str, embedding_extractor: str, to_save: bool=None):
    """
    Function that takes a YouTube video ID as input and returns the captions
    as a pandas DataFrame with the key "caption".
    :param video_id: YouTube video ID
    :param query: Query to answer
    :param max_tokens: Maximum number of tokens to use for the prompt
    :param model_lang: Language of the model to use for computing embeddings. Options are "en" and "tr"
    :param embedding_extractor: Extractor to use for computing embeddings. Options are "hf" and "openai"
    :param to_save: Whether to save the embeddings to MongoDB
    :return: Pandas DataFrame with the key "caption"
    """
    print("Extracting text...")
    if not video_id:
        raise ValueError("Video ID is missing")

    yt_sub_df = scrape_youtube(video_id)

    yt_sub_embeddings = None

    print("Computing embeddings...")

    if embedding_extractor == "hf":
        yt_sub_embeddings = compute_doc_embeddings_hf(yt_sub_df, model_lang)
    else:
        yt_sub_embeddings = compute_doc_embeddings(yt_sub_df)

    print("Answering query...")

    answer = ""
    if embedding_extractor == "hf":
        answer, prompt, messages  = answer_query_with_context(query, yt_sub_df, yt_sub_embeddings, embedding_type="hf", model_lang=model_lang, max_tokens=max_tokens)
    else:
        answer, prompt, messages  = answer_query_with_context(query, yt_sub_df, yt_sub_embeddings, embedding_type="openai", model_lang=model_lang, max_tokens=max_tokens)

    if to_save == "true":
        mongo_client.pair_yt.insert_one({"query": query, "answer": answer, "prompt": prompt})

    print("Done!")

    return answer, prompt, messages



