from knowledgegpt.extractors.base_extractor import BaseExtractor
from knowledgegpt.utils.utils_pdf import process_pdf, process_pdf_page
from knowledgegpt.utils.utils_powerpoint import process_pptx
from knowledgegpt.utils.utils_docs import extract_paragraphs

from io import BytesIO


class HybridFileExtractpr(BaseExtractor):
    def __init__(self, directory_path: str, extraction_type: str = "page", embedding_extractor: str = "hf",
                 model_lang: str = "en", is_turbo: bool = False, verbose: bool = False, index_path: str = None, index_type: str = "basic"):
        """
        Extracts paragraphs from a PDF file and computes embeddings for each paragraph,
        then answers a query using the embeddings.
        """
        super().__init__(embedding_extractor=embedding_extractor, model_lang=model_lang, is_turbo=is_turbo,
                         verbose=verbose, index_path=index_path, index_type=index_type)

        self.directory_path = directory_path
        self.extraction_type = extraction_type

    def prepare_df(self):
        if self.df is None:
            if not self.verbose:
                print("Processing PDF file...")
                print("Extracting paragraphs...")
            import os
            
            if  os.path.isdir(self.directory_path):
                import pandas as pd
                self.df = pd.DataFrame()

                pdf_files = [os.path.join(self.directory_path, f) for f in os.listdir(self.directory_path) if f.endswith(".pdf")]
                for pdf_file in pdf_files:
                    try:

                        if self.extraction_type == "page":
                            self.df = self.df.append(process_pdf_page(pdf_file))
                        else:
                            self.df = self.df.append(process_pdf(pdf_file))
                    except:
                        print("Error in file: ", pdf_file)
                        continue

                doc_files = [os.path.join(self.directory_path, f) for f in os.listdir(self.directory_path) if f.endswith(".doc") or f.endswith(".docx")]
                for doc_file in doc_files:
                    _, ext = os.path.splitext(doc_file)
                    allowed_ext = [".doc", ".docx"]
                    if ext not in allowed_ext:
                        return {"error": "Only Word files are allowed"}
                    try:

                        with open(doc_file, "rb") as f:
                            docs_buffer = BytesIO(f.read())

                        self.df = self.df.append(extract_paragraphs(docs_buffer))
                    except:
                        print("Error in file: ", doc_file)
                        continue

                pptx_files = [os.path.join(self.directory_path, f) for f in os.listdir(self.directory_path) if f.endswith(".pptx")]
                for pptx_file in pptx_files:
                    try:
                        with open(pptx_file, "rb") as f:
                            pptx_buffer = BytesIO(f.read())

                        self.df = self.df.append(process_pptx(pptx_buffer))
                    except:
                        print("Error in file: ", pptx_file)
                        continue

                self.df = self.df.reset_index()
