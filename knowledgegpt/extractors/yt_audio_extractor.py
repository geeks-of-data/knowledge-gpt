from typing import Optional
from ..utils.utils_yt_whisper import transcribe_youtube_audio
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context


def youtube_audio_scrape(mongo_client, video_id: str, query: str, max_tokens, model_lang: str, embedding_extractor: str, to_save: Optional[bool] = False):
    """
    Takes a YouTube video ID as input, transcribes its audio, and returns the 
    embeddings of the resulting text. Uses the embeddings to answer a query, 
    then saves the query and answer to a MongoDB database if specified.
    :param video_id: YouTube video ID
    :param max_tokens: Maximum number of tokens to use for the prompt
    :param query: Query to answer
    :param model_lang: Language of the model to use for computing embeddings. Options are "en" and "tr"
    :param embedding_extractor: Extractor to use for computing embeddings. Options are "hf" and "openai"
    :param to_save: Whether to save the embeddings to MongoDB
    :return: Answer to the query and the prompt and messages
    """

    print("Transcribing audio...")

    yt_audio_df = None
    yt_audio_embeddings = None

    if not video_id:
        raise ValueError("Video ID is missing")
    
    print("Extracting text...")

    if yt_audio_df is None:
        yt_audio_df = transcribe_youtube_audio(video_id)

    print("Computing embeddings...")

    if yt_audio_embeddings is None:
        if embedding_extractor == "hf":
           yt_audio_embeddings = compute_doc_embeddings_hf(yt_audio_df, model_lang)
        else:
           yt_audio_embeddings = compute_doc_embeddings(yt_audio_df)

    print("Answering query...")

    if embedding_extractor == "hf":
        answer, prompt, messages = answer_query_with_context(query, yt_audio_df, yt_audio_embeddings, embedding_type="hf", model_lang=model_lang, max_tokens=max_tokens)
    else:
        answer, prompt, messages = answer_query_with_context(query, yt_audio_df, yt_audio_embeddings, embedding_type="openai", model_lang=model_lang, max_tokens=max_tokens)

    if to_save:
        mongo_client.pair_yt_audio.insert_one({"query": query, "answer": answer, "prompt": prompt})

    print("Done!")

    return answer, prompt, messages
