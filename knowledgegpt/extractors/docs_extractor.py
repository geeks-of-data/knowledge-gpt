import os

from knowledgegpt.extractors.base_extractor import BaseExtractor

from io import BytesIO
from knowledgegpt.utils.utils_docs import extract_paragraphs


class DocsExtractor(BaseExtractor):
    """
    Extracts paragraphs from a Word file and computes embeddings for each paragraph, then answers a query using the
    embeddings.
    """

    def __init__(self, file_path: str, embedding_extractor: str = "hf", model_lang: str = "en", is_turbo: bool = False,
                 verbose: bool = False, index_path: str = None, index_type: str = "basic"):
        super().__init__(embedding_extractor=embedding_extractor, model_lang=model_lang, is_turbo=is_turbo,
                         verbose=verbose, index_path=index_path, index_type=index_type)
        self.file_path = file_path


    def prepare_df(self):
        if self.df is None:
            if not self.verbose:
                print("Extracting paragraphs...")
            import os

            if os.path.isdir(self.file_path):
                import pandas as pd
                files = [os.path.join(self.file_path, f) for f in os.listdir(self.file_path) if f.endswith(".doc") or f.endswith(".docx")]
                self.df = pd.DataFrame()
                for file in files:
                    _, ext = os.path.splitext(file)
                    allowed_ext = [".doc", ".docx"]
                    if ext not in allowed_ext:
                        return {"error": "Only Word files are allowed"}

                    with open(file, "rb") as f:
                        docs_buffer = BytesIO(f.read())

                    self.df = self.df.append(extract_paragraphs(docs_buffer))

                self.df = self.df.reset_index()

            else:
                if not os.path.isfile(self.file_path):
                    return {"error": "File not found"}

                _, ext = os.path.splitext(self.file_path)
                allowed_ext = [".doc", ".docx"]
                if ext not in allowed_ext:
                    return {"error": "Only Word files are allowed"}

                with open(self.file_path, "rb") as f:
                    docs_buffer = BytesIO(f.read())

                self.df = extract_paragraphs(docs_buffer)
