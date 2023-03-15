from ..utils.utils_powerpoint import process_pptx
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context
from io import BytesIO
import os

class PowerpointExtractor:
    def __init__(self, file_path,   embedding_extractor: str = "hf", model_lang: str = "en"):
        self.file_path = file_path
        self.model_lang = model_lang
        self.embedding_extractor = embedding_extractor
        self.max_tokens = 1000
        self.to_save = False
        self.mongo_client = None
        self.df = None
        self.embeddings = None
        

    def extract(self,   query: str = "",max_tokens: int = 1000,to_save: bool = False, mongo_client = None):
        """
        Function that takes a file path for a PowerPoint presentation as input and returns the response answer.

        :param file_path: Path to the PowerPoint file
        :param query: Query to answer
        :return: Answer to the query and the prompt and messages
        """
        print("Processing PowerPoint file...")

        if max_tokens is not None:
            self.max_tokens = max_tokens

        if not os.path.isfile(self.file_path):
            raise ValueError("Invalid file path provided.")

        if not self.file_path.endswith(".pptx"):
            raise ValueError("Only PowerPoint (.pptx) files are allowed.")


        print("Extracting paragraphs...")

        with open(self.file_path, "rb") as f:
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

        if to_save:
            print("Saving to MongoDB...")
            self.mongo_client = mongo_client
            self.mongo_client.pair_pptx.insert_one({"query": target, "answer": answer, "prompt": prompt})

        print("Done!")

        return answer, prompt, messages
