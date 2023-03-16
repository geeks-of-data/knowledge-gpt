from ..utils.utils_pdf import process_pdf, process_pdf_page
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context
from io import BytesIO
from typing import Optional, Tuple, List

class PDFExtractor:
    def __init__(self, pdf_file_path: str, extraction_type: str = "page", embedding_extractor: str = "hf", model_lang: str = "en",is_turbo: bool = False):
        """
        Extracts paragraphs from a PDF file and computes embeddings for each paragraph, then answers a query using the embeddings.
        :param extraction_type: Type of extraction to use. Options are "page" and "paragraph"
        :param embedding_extractor: Extractor to use for computing embeddings. Options are "hf" and "openai"
        :param model_lang: Language of the model to use for computing embeddings. Options are "en" and "tr"
        :param to_save: Whether to save the embeddings to MongoDB
        :param max_tokens: Maximum number of tokens to use for the answer
        :param mongo_client: MongoDB client
        """
        self.mongo_client = None
        self.max_tokens = 1000
        self.is_turbo = is_turbo
        self.pdf_file_path =pdf_file_path
        self.extraction_type = extraction_type
        self.embedding_extractor = embedding_extractor
        self.model_lang = model_lang
        self.df = None
        self.embeddings = None
        self.messages = []
        self.is_first_time = True
    
    def extract(self, query: str,max_tokens=None, to_save=False,  mongo_client=None ) -> Tuple[str, Optional[str], List[str]]:
        """
        Extracts paragraphs from a PDF file and computes embeddings for each paragraph, then answers a query using the embeddings.
        :param query: Query to answer
        :param pdf_file_path: Path to the PDF file
        :return: Answer to the query and the prompt and messages
        """
        print("Processing PDF file...")

        if max_tokens is not None:
            self.max_tokens = max_tokens

        if self.is_first_time:

            with open(self.pdf_file_path, "rb") as f:
                pdf_file = BytesIO(f.read())

            if pdf_file.getvalue()[:4] != b'%PDF':
                raise ValueError("Only PDF files are allowed")

        print("Extracting paragraphs...")

        if self.df is None:
            if self.extraction_type == "page":
                self.df = process_pdf_page(pdf_file)
            else:
                self.df = process_pdf(pdf_file)

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

        target = query
        answer = ""
        if self.embedding_extractor == "hf":
            answer, prompt, self.messages = answer_query_with_context(target, self.df, self.embeddings, embedding_type="hf", model_lang=self.model_lang, is_turbo=self.is_turbo, messages=self.messages, is_first_time=self.is_first_time, max_tokens=max_tokens)
        else:
            answer, prompt, self.messages = answer_query_with_context(target, self.df, self.embeddings, embedding_type="openai", model_lang=self.model_lang, is_turbo=self.is_turbo, messages=self.messages, is_first_time=self.is_first_time, max_tokens=max_tokens)

        if to_save:
            print("Saving to MongoDB...")
            self.mongo_client = mongo_client
            self.mongo_client.pair_pdf.insert_one({"query": target, "answer": answer, "prompt": prompt})

        print("Done!")

        return answer, prompt, self.messages