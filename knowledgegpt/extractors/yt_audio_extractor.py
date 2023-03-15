from typing import Optional
from ..utils.utils_yt_whisper import transcribe_youtube_audio
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context

class YoutubeAudioExtractor:
    def __init__(self, mongo_client, model_lang='en', embedding_extractor='hf'):
        self.mongo_client = mongo_client
        self.model_lang = model_lang
        self.embedding_extractor = embedding_extractor

        self.df = None
        self.embeddings = None

    def extract(self, video_id: str, query: str, max_tokens, to_save: Optional[bool] = False):
        """
        Takes a YouTube video ID as input, transcribes its audio, and returns the 
        embeddings of the resulting text. Uses the embeddings to answer a query, 
        then saves the query and answer to a MongoDB database if specified.
        :param video_id: YouTube video ID
        :param max_tokens: Maximum number of tokens to use for the prompt
        :param query: Query to answer
        :param to_save: Whether to save the embeddings to MongoDB
        :return: Answer to the query and the prompt and messages
        """
        print("Transcribing audio...")


        if not video_id:
            raise ValueError("Video ID is missing")

        print("Extracting text...")

        if self.df is None:
            self.df = transcribe_youtube_audio(video_id)

        print("Computing embeddings...")

        if self.embeddings is None:
            if self.embedding_extractor == "hf":
               self.embeddings = compute_doc_embeddings_hf(self.df, self.model_lang)
            else:
               self.embeddings = compute_doc_embeddings(self.df)

        print("Answering query...")

        if self.embedding_extractor == "hf":
            answer, prompt, messages = answer_query_with_context(query, self.df, self.embeddings, embedding_type="hf", model_lang=self.model_lang, max_tokens=max_tokens)
        else:
            answer, prompt, messages = answer_query_with_context(query, self.df, self.embeddings, embedding_type="openai", model_lang=self.model_lang, max_tokens=max_tokens)

        if to_save:
            self.mongo_client.pair_yt_audio.insert_one({"query": query, "answer": answer, "prompt": prompt})

        print("Done!")

        return answer, prompt, messages