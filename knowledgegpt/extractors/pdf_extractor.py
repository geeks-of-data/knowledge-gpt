from ..utils.utils_pdf import process_pdf, process_pdf_page
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context
from io import BytesIO
from typing import Optional, Tuple, List

class PDFExtractor:
    def __init__(self, mongo_client, max_tokens, extraction_type: str = "page", embedding_extractor: str = "hf", model_lang: str = "en", to_save: bool = False):
        """
        Extracts paragraphs from a PDF file and computes embeddings for each paragraph, then answers a query using the embeddings.
        :param extraction_type: Type of extraction to use. Options are "page" and "paragraph"
        :param embedding_extractor: Extractor to use for computing embeddings. Options are "hf" and "openai"
        :param model_lang: Language of the model to use for computing embeddings. Options are "en" and "tr"
        :param to_save: Whether to save the embeddings to MongoDB
        :param max_tokens: Maximum number of tokens to use for the answer
        :param mongo_client: MongoDB client
        """
        self.mongo_client = mongo_client
        self.max_tokens = max_tokens
        self.extraction_type = extraction_type
        self.embedding_extractor = embedding_extractor
        self.model_lang = model_lang
        self.to_save = to_save
        self.pdf_df = None
        self.pdf_embeddings = None
    
    def extract(self, query: str, pdf_file_path: str) -> Tuple[str, Optional[str], List[str]]:
        """
        Extracts paragraphs from a PDF file and computes embeddings for each paragraph, then answers a query using the embeddings.
        :param query: Query to answer
        :param pdf_file_path: Path to the PDF file
        :return: Answer to the query and the prompt and messages
        """
        print("Processing PDF file...")

        with open(pdf_file_path, "rb") as f:
            pdf_file = BytesIO(f.read())

        if pdf_file.getvalue()[:4] != b'%PDF':
            raise ValueError("Only PDF files are allowed")

        print("Extracting paragraphs...")

        if self.pdf_df is None:
            if self.extraction_type == "page":
                self.pdf_df = process_pdf_page(pdf_file)
            else:
                self.pdf_df = process_pdf(pdf_file)

        print("Computing embeddings...")

        if self.pdf_embeddings is None:
            if self.embedding_extractor == "hf":
                self.pdf_embeddings = compute_doc_embeddings_hf(self.pdf_df, self.model_lang)
            else:
                self.pdf_embeddings = compute_doc_embeddings(self.pdf_df)

        print("Answering query...")

        target = query
        answer = ""
        if self.embedding_extractor == "hf":
            answer, prompt, messages = answer_query_with_context(target, self.pdf_df, self.pdf_embeddings, embedding_type="hf", model_lang=self.model_lang, max_tokens=self.max_tokens)
        else:
            answer, prompt, messages = answer_query_with_context(target, self.pdf_df, self.pdf_embeddings, embedding_type="openai", model_lang=self.model_lang, max_tokens=self.max_tokens)

        if self.to_save:
            self.mongo_client.pair_pdf.insert_one({"query": target, "answer": answer, "prompt": prompt})

        print("Done!")

        return answer, prompt, messages