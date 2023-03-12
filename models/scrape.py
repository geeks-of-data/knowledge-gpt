from pydantic import BaseModel

class ScrapeQuery(BaseModel):
    url: str
    embedding_extractor: str
    model_lang: str
    query: str