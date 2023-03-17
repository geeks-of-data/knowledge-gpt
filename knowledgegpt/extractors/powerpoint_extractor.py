import os
from knowledgegpt.extractors.base_extractor import BaseExtractor

from knowledgegpt.utils.utils_powerpoint import process_pptx
from io import BytesIO


class PowerpointExtractor(BaseExtractor):
    """
    Function that takes a file path for a PowerPoint presentation as input and returns the response answer.
    """

    def __init__(self, file_path, embedding_extractor: str = "hf", model_lang: str = "en", is_turbo: bool = False):

        super().__init__(embedding_extractor, model_lang, is_turbo)
        self.file_path = file_path

    def prepare_df(self):
        if self.df is None:
            if not self.verbose:
                print("Extracting paragraphs...")

            if not os.path.isfile(self.file_path):
                raise ValueError("Invalid file path provided.")

            if not self.file_path.endswith(".pptx"):
                raise ValueError("Only PowerPoint (.pptx) files are allowed.")
            with open(self.file_path, "rb") as f:
                pptx_buffer = BytesIO(f.read())

            self.df = process_pptx(pptx_buffer)
