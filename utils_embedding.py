from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np

sentence_tf_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_hf_embeddings(text: str, model=sentence_tf_model) -> np.ndarray:

    sentence_embeddings = model.encode(text)
    sentence_embeddings = sentence_embeddings.reshape(1, -1)
    sentence_embeddings = normalize(sentence_embeddings)
    return sentence_embeddings[0]

def compute_doc_embeddings_hf(df: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:

    return {
        idx: get_hf_embeddings(r.content, sentence_tf_model) for idx, r in df.iterrows()
    }
    