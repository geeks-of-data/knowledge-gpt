from knowledgegpt.extractors.helpers import check_embedding_extractor
from knowledgegpt.utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from knowledgegpt.utils.utils_completion import answer_query_with_context


class BasicExtractor:
    def __init__(self, dataframe, embedding_extractor="hf", model_lang="en", is_turbo=False, index_type="basic"):
        self.df = dataframe
        check_embedding_extractor(
            embedding_extractor=embedding_extractor
        )
        self.embedding_extractor = embedding_extractor
        self.model_lang = model_lang
        self.is_turbo = is_turbo
        self.messages = []
        self.is_first_time = True
        self.index_type = index_type

        self.to_save = False
        self.mongo_client = None
        self.embeddings = None
        self.answer = ""
        self.prompt = ""

    def extract(self, query, max_tokens, to_save=False, mongo_client=None):

        print("Processing dataframe...")

        print("Computing embeddings...")

        if self.embeddings is None:
            if self.embedding_extractor == "hf":
                embeddings = compute_doc_embeddings_hf(self.df, self.model_lang)
            else:
                embeddings = compute_doc_embeddings(self.df)
            self.embeddings = embeddings

        print("Answering query...")

        self.answer, self.prompt, self.messages = answer_query_with_context(
            query=query,
            df=self.df,
            document_embeddings=self.embeddings,
            embedding_type=self.embedding_extractor,
            model_lang=self.model_lang,
            is_turbo=self.is_turbo,
            messages=self.messages,
            is_first_time=self.is_first_time,
            max_tokens=max_tokens,
            index_type=self.index_type
        )

        if to_save:
            print("Saving to MongoDB...")
            self.mongo_client = mongo_client
            self.mongo_client.pair_pdf.insert_one({"query": query, "answer": self.answer, "prompt": self.prompt})

        print("AllDone!")

        return self.answer, self.prompt, self.messages
