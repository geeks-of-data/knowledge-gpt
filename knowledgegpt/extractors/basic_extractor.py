from typing import List
import pandas as pd
from ..utils.utils_docs import extract_paragraphs
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context

class BasicExtractor:
    def __init__(self, mongo_client, dataframe, embedding_extractor="hf", model_lang="en", to_save=False, is_turbo=False, messages=None):
        self.mongo_client = mongo_client
        self.df = dataframe
        self.embedding_extractor = embedding_extractor
        self.model_lang = model_lang
        self.to_save = to_save
        self.is_turbo = is_turbo
        self.messages = messages
        self.embeddings = None

    def extract(self, query, max_tokens):
        print("Processing dataframe...")

        embeddings = self.embeddings

        print("Computing embeddings...")

        if embeddings is None:
            if self.embedding_extractor == "hf":
                embeddings = compute_doc_embeddings_hf(self.df, self.model_lang)
            else:
                embeddings = compute_doc_embeddings(self.df)
            self.embeddings = embeddings

        print("Answering query...")

        target = query
        answer = ""
        if self.embedding_extractor == "hf":
            answer, prompt, messages = answer_query_with_context(target, self.df, embeddings, embedding_type="hf", model_lang=self.model_lang, max_tokens=max_tokens)
        else:
            answer, prompt, messages = answer_query_with_context(target, self.df, embeddings, embedding_type="openai", model_lang=self.model_lang, max_tokens=max_tokens)
        self.messages = messages

        if self.to_save:
            self.mongo_client.pair_pdf.insert_one({"query": target, "answer": answer, "prompt": prompt})

        print("Done!")

        return answer, prompt, self.messages
