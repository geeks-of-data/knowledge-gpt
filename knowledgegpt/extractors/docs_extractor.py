import os

from io import BytesIO

from knowledgegpt.extractors.helpers import check_embedding_extractor
from knowledgegpt.utils.utils_docs import extract_paragraphs
from knowledgegpt.utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from knowledgegpt.utils.utils_completion import answer_query_with_context


class DocsExtractor:
    def __init__(self, file_path: str, embedding_extractor: str = "hf", model_lang: str = "en", is_turbo: bool = False):
        self.file_path = file_path
        check_embedding_extractor(
            embedding_extractor=embedding_extractor
        )
        self.embedding_extractor = embedding_extractor
        self.model_lang = model_lang
        self.is_turbo = is_turbo
        self.mongo_client = None
        self.df = None
        self.embeddings = None
        self.messages = []
        self.is_first_time = True
        self.answer = ""
        self.prompt = ""

    def extract(self, query: str, max_tokens, to_save: bool = False, mongo_client=None) -> dict:
        """
        Extracts paragraphs from a Word file and computes embeddings for each paragraph, then answers a query using the
        embeddings.
        :param query: Query to answer
        :param max_tokens: Maximum number of tokens to generate
        :param to_save: Whether to save the embeddings to MongoDB
        :param mongo_client: If to_save true then needed mongo client else do not require any mongo_client
        :return: Answer to the query and the prompt and messages
        """
        print("Processing Word file...")

        if not self.is_first_time:
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

        print("Answering query...")

        if len(self.messages) == 0 and self.is_turbo == True:
            self.messages = [{"role": "system", "content": "you are a helpful assistant"}]

        if len(self.messages) > 2:
            self.is_first_time = False
            print("not the first time")

        self.answer, self.prompt, self.messages = answer_query_with_context(
            query=query,
            df=self.df,
            document_embeddings=self.embeddings,
            embedding_type=self.embedding_extractor,
            model_lang=self.model_lang,
            is_turbo=self.is_turbo,
            messages=self.messages,
            is_first_time=self.is_first_time,
            max_tokens=max_tokens
        )

        if to_save:
            # if to_save true then insert into mongo else do not required mongo client or pymongo etc.
            print("Saving to Mongo...")
            self.mongo_client = mongo_client
            if not self.is_turbo:
                self.mongo_client.pairs_docs.insert_one({
                    "query": query,
                    "answer": self.answer,
                    "prompt": self.prompt
                })
            else:
                self.mongo_client.pairs_docs_turbo.insert_one({"conversation": self.messages})

        print("AllDone!")

        return self.answer, self.prompt, self.messages
