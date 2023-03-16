from typing import List
import os
from ..utils.utils_docs import extract_paragraphs
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context
from io import BytesIO

class DocsExtractor:
    def __init__(self,file_path: str,  embedding_extractor: str = "hf", model_lang: str = "en",  is_turbo: bool = False):
        self.file_path=file_path
        self.embedding_extractor = embedding_extractor
        self.model_lang = model_lang
        self.is_turbo = is_turbo
        self.mongo_client = None
        self.df = None
        self.embeddings = None
        self.messages = []
        self.is_first_time = True
        
    def extract(self, query: str, max_tokens, to_save: bool = False,mongo_client=None) -> dict:
        """
        Extracts paragraphs from a Word file and computes embeddings for each paragraph, then answers a query using the embeddings.
        :param file_path: Path to the Word file
        :param query: Query to answer
        :param max_tokens: Maximum number of tokens to generate
        :param embedding_extractor: Extractor to use for computing embeddings. Options are "hf" and "openai"
        :param model_lang: Language of the model to use for computing embeddings. Options are "en" and "tr"
        :param to_save: Whether to save the embeddings to MongoDB
        :param is_turbo: Whether to use turbo mode
        :return: Answer to the query and the prompt and messages
        """
        print("Processing Word file...")

        if self.is_first_time == False:

            if not os.path.isfile(self.file_path):
                return {"error": "File not found"}

            _, ext = os.path.splitext(self.file_path)
            allowed_ext = [".doc", ".docx"]
            if ext not in allowed_ext:
                return {"error": "Only Word files are allowed"}

        print("Extracting paragraphs...")

        if self.df is None:

            with open(self.file_path, "rb") as f:
                docs_buffer = BytesIO(f.read())

            self.df = extract_paragraphs(docs_buffer)

        print("Computing embeddings...")

        if self.embeddings is None:
            if self.embedding_extractor == "hf":
                self.embeddings = compute_doc_embeddings_hf(self.df, self.model_lang)
            else:
                self.embeddings = compute_doc_embeddings(self.df)

        target = query
        answer = ""

        print("Answering query...")

        if len(self.messages) == 0 and self.is_turbo == True:
            self.messages = [{"role": "system", "content": "you are a helpful assistant"}]

        if len(self.messages) > 2:
            self.is_first_time = False
            print("not the first time")

        if self.embedding_extractor == "hf":
            answer, prompt, self.messages = answer_query_with_context(target, self.df, self.embeddings, embedding_type="hf", model_lang=self.model_lang, is_turbo=self.is_turbo, messages=self.messages, is_first_time=self.is_first_time, max_tokens=max_tokens)
        else:
            answer, prompt, self.messages = answer_query_with_context(target, self.df, self.embeddings, embedding_type="openai", model_lang=self.model_lang, is_turbo=self.is_turbo, messages=self.messages, is_first_time=self.is_first_time, max_tokens=max_tokens)

        if to_save == True:
            print("Saving to Mongo...")
            self.mongo_client = mongo_client
            if  self.is_turbo == False:
                self.mongo_client.pairs_docs.insert_one({"query": target, "answer": answer, "prompt": prompt})
            else:
                self.mongo_client.pairs_docs_turbo.insert_one({"conversation": self.messages})

        print("Done!")

        return answer, prompt, self.messages
