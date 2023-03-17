from knowledgegpt.extractors.helpers import check_embedding_extractor
from knowledgegpt.utils.utils_powerpoint import process_pptx
from knowledgegpt.utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from knowledgegpt.utils.utils_completion import answer_query_with_context
from io import BytesIO
import os


class PowerpointExtractor:
    def __init__(self, file_path, embedding_extractor: str = "hf", model_lang: str = "en", is_turbo: bool = False):
        self.file_path = file_path
        check_embedding_extractor(
            embedding_extractor=embedding_extractor
        )
        self.model_lang = model_lang
        self.embedding_extractor = embedding_extractor
        self.max_tokens = 1000
        self.to_save = False
        self.mongo_client = None
        self.df = None
        self.embeddings = None
        self.is_first_time = True
        self.messages = []
        self.is_turbo = is_turbo
        self.answer = ""
        self.prompt = ""

    def extract(self, query: str = "", max_tokens: int = 1000, to_save: bool = False, mongo_client=None):
        """
        Function that takes a file path for a PowerPoint presentation as input and returns the response answer.

        :param file_path: Path to the PowerPoint file
        :param query: Query to answer
        :return: Answer to the query and the prompt and messages
        """
        print("Processing PowerPoint file...")

        if max_tokens is not None:
            self.max_tokens = max_tokens

        if self.is_first_time:

            if not os.path.isfile(self.file_path):
                raise ValueError("Invalid file path provided.")

            if not self.file_path.endswith(".pptx"):
                raise ValueError("Only PowerPoint (.pptx) files are allowed.")

        print("Extracting paragraphs...")

        if self.df is None:
            with open(self.file_path, "rb") as f:
                pptx_buffer = BytesIO(f.read())

            self.df = process_pptx(pptx_buffer)

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
            print("Saving to MongoDB...")
            self.mongo_client = mongo_client
            self.mongo_client.pair_pptx.insert_one({"query": query, "answer": self.answer, "prompt": self.prompt})

        print("Done!")

        return self.answer, self.prompt, self.messages
