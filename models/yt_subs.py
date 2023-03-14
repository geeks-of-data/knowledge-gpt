from pydantic import BaseModel

class YTSubsQuery(BaseModel):
    video_id: str
    query: str
    model_lang: str
    embedding_extractor: str
    to_save: str
