def check_embedding_extractor(embedding_extractor, embedding_extractor_acceptable_list=None):
    if embedding_extractor_acceptable_list is None:
        embedding_extractor_acceptable_list = ["hf", "openai"]

    if not isinstance(embedding_extractor, str):
        raise Exception("Embedding Extractor must be a string")

    if embedding_extractor not in embedding_extractor_acceptable_list:
        raise Exception(f"Embedding Extractor is not allowed. "
                        f"Please choose one of : {embedding_extractor_acceptable_list}")


def check_model_lang(model_lang, model_lang_acceptable_list=None):
    if model_lang_acceptable_list is None:
        model_lang_acceptable_list = ["en", "tr"]

    if not isinstance(model_lang, str):
        raise Exception("Model Lang must be a string")

    if model_lang not in model_lang_acceptable_list:
        raise Exception(f"Model Lang is not allowed. "
                        f"Please choose one of : {model_lang_acceptable_list}")


def check_index_type(index_type, index_type_acceptable_list=None):
    if index_type_acceptable_list is None:
        index_type_acceptable_list = ["basic", "faiss"]

    if not isinstance(index_type, str):
        raise Exception("Index Type must be a string")

    if index_type not in index_type_acceptable_list:
        raise Exception(f"Index Type is not allowed. "
                        f"Please choose one of : {index_type_acceptable_list}")
