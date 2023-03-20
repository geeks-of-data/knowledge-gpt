from knowledgegpt.extractors.base_extractor import BaseExtractor
from knowledgegpt.utils.utils_yt_whisper import transcribe_youtube_audio
from knowledgegpt.utils.utils_yt_playlist import yt_playlist_link_parser


class YoutubeAudioExtractor(BaseExtractor):
    """
    Takes a YouTube video ID as input, transcribes its audio, and returns the
    embeddings of the resulting text. Uses the embeddings to answer a query.
    """

    def __init__(self, video_id: str, embedding_extractor='hf', model_lang='en', is_turbo: bool = False,
                 verbose: bool = False, index_path: str = None, index_type: str = "basic", is_playlist: bool = False):
        super().__init__(embedding_extractor=embedding_extractor, model_lang=model_lang, is_turbo=is_turbo,
                         verbose=verbose, index_path=index_path, index_type=index_type)
        self.video_id = video_id
        self.is_playlist = is_playlist

    def prepare_df(self):
        if self.df is None:
            
            if not self.verbose:
                print("Transcribing audio...")

            if not self.video_id:
                raise ValueError("Video ID is missing")
            
            if self.is_playlist:
                import pandas as pd
                video_links = yt_playlist_link_parser(self.video_id)

                self.df = pd.DataFrame()
                for video_id in video_links:
                    self.df = self.df.append(transcribe_youtube_audio(video_id))

                self.df = self.df.reset_index()
            else:    
                self.df = transcribe_youtube_audio(self.video_id)
