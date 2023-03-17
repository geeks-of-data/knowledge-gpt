from knowledgegpt.extractors.helpers import check_embedding_extractor
from knowledgegpt.utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from knowledgegpt.utils.utils_completion import answer_query_with_context


class BaseExtractor:
    def __init__(self, dataframe=None, embedding_extractor="hf", model_lang="en", is_turbo=False, index_type="basic",
                 verbose=False):
        """
        :param dataframe: if you have own df use it else choose correct extractor
        :param embedding_extractor: default hf, openai
        :param model_lang: default en
        :param is_turbo: default False
        :param index_type: default basic
        """
        check_embedding_extractor(
            embedding_extractor=embedding_extractor
        )
        self.embedding_extractor = embedding_extractor
        self.model_lang = model_lang
        self.is_turbo = is_turbo
        self.index_type = index_type
        self.verbose = verbose
        self.messages = []

        self.embeddings = None
        self.answer = ""
        self.prompt = ""
        self.df = dataframe

    def prepare_df(self):
        pass

    def set_embeddings(self):
        if self.embeddings is None:
            if not self.verbose:
                print("Computing embeddings...")
            if self.embedding_extractor == "hf":
                self.embeddings = compute_doc_embeddings_hf(self.df, self.model_lang)
            else:
                self.embeddings = compute_doc_embeddings(self.df)

    def extract(self, query, max_tokens) -> tuple[str, str, list]:
        """
        param query: Query to answer
        param max_tokens: Maximum number of tokens to generate
        """
        self.prepare_df()
        self.set_embeddings()

        if len(self.messages) == 0 and self.is_turbo:
            self.messages = [{"role": "system", "content": "you are a helpful assistant"}]

        self.answer, self.prompt, self.messages = answer_query_with_context(
            query=query,
            df=self.df,
            document_embeddings=self.embeddings,
            embedding_type=self.embedding_extractor,
            model_lang=self.model_lang,
            is_turbo=self.is_turbo,
            verbose=self.verbose,
            messages=self.messages,
            max_tokens=max_tokens,
            index_type=self.index_type
        )
        if not self.verbose:
            print("all_done!")
        return self.answer, self.prompt, self.messages
