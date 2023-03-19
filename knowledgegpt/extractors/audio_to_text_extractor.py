from knowledgegpt.extractors.base_extractor import BaseExtractor
from knowledgegpt.utils.utils_raw_whisper import transcribe_audio


class AudioToTextExtractor(BaseExtractor):
    """
    Takes an audio file as input, transcribes it, and returns the
    embeddings of the resulting text. Uses the embeddings to answer a query.
    """

    def __init__(self, audio_path: str, embedding_extractor='hf', model_lang='en', is_turbo: bool = False,
                 verbose: bool = False, index_path: str = None, index_type: str = "basic"):
        super().__init__(embedding_extractor=embedding_extractor, model_lang=model_lang, is_turbo=is_turbo,
                         verbose=verbose, index_path=index_path, index_type=index_type)
        self.audio_path = audio_path

    def prepare_df(self):

        import pandas as pd

        if self.df is None:
            if not self.verbose:
                print("Transcribing audio...")
            if not self.audio_path:
                raise ValueError("Audio path is missing")
            parahraphs = transcribe_audio(self.audio_path)
            self.df = pd.DataFrame({"content": parahraphs})
