import os
from knowledgegpt.extractors.base_extractor import BaseExtractor

from knowledgegpt.utils.utils_powerpoint import process_pptx
from io import BytesIO


class PowerpointExtractor(BaseExtractor):
    """
    Function that takes a file path for a PowerPoint presentation as input and returns the response answer.
    """

    def __init__(self, file_path, embedding_extractor: str = "hf", model_lang: str = "en", is_turbo: bool = False,
                 verbose: bool = False, index_path: str = None, index_type: str = "basic"):

        super().__init__(embedding_extractor=embedding_extractor, model_lang=model_lang, is_turbo=is_turbo,
                         verbose=verbose, index_path=index_path, index_type=index_type)
        self.file_path = file_path


    def prepare_df(self):
        if self.df is None:
            if not self.verbose:
                print("Extracting paragraphs...")
            import os
            
            if  os.path.isdir(self.file_path):
                import pandas as pd
                pptx_files = [os.path.join(self.file_path, f) for f in os.listdir(self.file_path) if f.endswith(".pptx")]
                self.df = pd.DataFrame()
                for pptx_file in pptx_files:
                    with open(pptx_file, "rb") as f:
                        pptx_buffer = BytesIO(f.read())

                    self.df = self.df.append(process_pptx(pptx_buffer))

                self.df = self.df.reset_index()

            else:
                if not os.path.isfile(self.file_path):
                    raise ValueError("Invalid file path provided.")

                if not self.file_path.endswith(".pptx"):
                    raise ValueError("Only PowerPoint (.pptx) files are allowed.")
                
                with open(self.file_path, "rb") as f:
                    pptx_buffer = BytesIO(f.read())

                self.df = process_pptx(pptx_buffer)
