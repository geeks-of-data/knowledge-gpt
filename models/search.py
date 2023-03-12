from pydantic import BaseModel

class SearchQuery(BaseModel):
    query: str
    embedding_extractor: str
    model_lang: str

# class EmbeddingExtractor(BaseModel):
#     embedding_extractor: str