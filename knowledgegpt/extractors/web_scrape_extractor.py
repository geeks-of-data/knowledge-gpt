from typing import Optional
from ..utils.utils_scrape import scrape_content
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context



class WebScrapeExtractor:
    def __init__(self, url, embedding_extractor: str, model_lang: str, is_turbo: bool = False):
        self.url = url
        self.embedding_extractor = embedding_extractor
        self.model_lang = model_lang
        self.max_tokens = 1000
        self.mongo_client = None
        self.df = None
        self.embeddings = None
        self.is_turbo = is_turbo
        self.messages = []
        self.is_first_time = True

    def extract(self,  query: Optional[str] = None, max_tokens: int=1000, to_save: bool = False, mongo_client=None):
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

        if self.is_first_time:

            if not self.url:
                raise ValueError("url is missing")

            if max_tokens is not None:
                self.max_tokens = max_tokens

        if self.df is None:

            self.df = scrape_content(self.url)

        print("Computing embeddings...")

        if self.embeddings is None:
            if self.embedding_extractor == "hf":
                self.embeddings = compute_doc_embeddings_hf(self.df, self.model_lang)
            else:
                self.embeddings = compute_doc_embeddings(self.df)

        if len(self.messages) == 0 and self.is_turbo == True:
            self.messages = [{"role": "system", "content": "you are a helpful assistant"}]

        if len(self.messages) > 2:
            self.is_first_time = False
            print("not the first time")

        target = query
        answer = ""

        print("Answering query...")

        if self.embedding_extractor == "hf":
            answer, prompt, self.messages = answer_query_with_context(target, self.df, self.embeddings, embedding_type="hf", model_lang=self.model_lang, is_turbo=self.is_turbo, messages=self.messages, is_first_time=self.is_first_time, max_tokens=max_tokens)
        else:
            answer, prompt, self.messages = answer_query_with_context(target, self.df, self.embeddings, embedding_type="openai", model_lang=self.model_lang, is_turbo=self.is_turbo, messages=self.messages, is_first_time=self.is_first_time, max_tokens=max_tokens)

        if to_save:
            print("Saving to Mongo...")
            self.mongo_client = mongo_client
            self.mongo_client.pair_web.insert_one({"query": target, "answer": answer, "prompt": prompt})

        print("Done!")

        return answer, prompt, self.messages