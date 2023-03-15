from ..utils.utils_powerpoint import process_pptx
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context
from io import BytesIO
import os

class PowerpointExtractor:
    def __init__(self, mongo_client, model_lang: str = "en", embedding_extractor: str = "hf", max_tokens: int = 64, to_save: bool = False):
        self.mongo_client = mongo_client
        self.model_lang = model_lang
        self.embedding_extractor = embedding_extractor
        self.max_tokens = max_tokens
        self.to_save = to_save
        self.df = None
        self.embeddings = None
        

    def extract(self, file_path: str, query: str = ""):
        """
        Function that takes a file path for a PowerPoint presentation as input and returns the response answer.

        :param file_path: Path to the PowerPoint file
        :param query: Query to answer
        :return: Answer to the query and the prompt and messages
        """
        print("Processing PowerPoint file...")

        if not os.path.isfile(file_path):
            raise ValueError("Invalid file path provided.")

        if not file_path.endswith(".pptx"):
            raise ValueError("Only PowerPoint (.pptx) files are allowed.")


        print("Extracting paragraphs...")

        with open(file_path, "rb") as f:
            pptx_buffer = BytesIO(f.read())

        self.df = process_pptx(pptx_buffer)

        print("Computing embeddings...")

        if self.embeddings is None:
            if self.embedding_extractor == "hf":
                self.embeddings = compute_doc_embeddings_hf(self.df, self.model_lang)
            else:
                self.embeddings = compute_doc_embeddings(self.df)

        target = query
        answer = ""

        print("Answering query...")

        if self.embedding_extractor == "hf":
            answer, prompt, messages = answer_query_with_context(target, self.df, self.embeddings, embedding_type="hf", model_lang=self.model_lang, max_tokens=self.max_tokens)
        else:
            answer, prompt, messages = answer_query_with_context(target, self.df, self.embeddings, embedding_type="openai", model_lang=self.model_lang, max_tokens=self.max_tokens)

        if self.to_save:
            self.mongo_client.pair_pptx.insert_one({"query": target, "answer": answer, "prompt": prompt})

        print("Done!")

        return answer, prompt, messages
