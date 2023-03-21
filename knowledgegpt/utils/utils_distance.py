# https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb sourced from here
import numpy as np
from knowledgegpt.utils.utils_embedding import get_hf_embeddings, get_embedding


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    :param x: The first vector.
    :param y: The second vector.
    :return: The similarity between the two vectors.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array], verbose=False,
                                                embedding_type: str = "hf", model_lang: str = 'en',
                                                index_type: str = "basic") -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    :param query: The query to answer.
    :param contexts: The embeddings of the document sections.
    :param embedding_type: The type of embedding used. Can be "hf" or "tf".
    :param model_lang: The language of the model. Can be "en" or "tr".
    :return: The list of document sections, sorted by relevance in descending order.
    """
    if not verbose:
        print("model_lang", model_lang)
    if embedding_type == "hf":
        query_embedding = get_hf_embeddings(query, model_lang)
    else:
        query_embedding = get_embedding(query)

    if index_type == "basic":

        document_similarities = sorted([
            (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in
            contexts.items()
        ], reverse=True)
    else:
        import faiss

        if embedding_type == "hf":
            dim = 384
        else:
            dim = 1536

        index = faiss.IndexFlatIP(dim)  # build the index

        # add vectors to the index
        index.add(np.array(list(contexts.values())))

        # query
        if embedding_type == "hf":
            query_embedding = query_embedding.reshape(1, dim)
        else:
            query_embedding = np.array(query_embedding)
            query_embedding = query_embedding.reshape(1, dim)

        D, I = index.search(query_embedding, len(contexts))  # actual search

        document_similarities = [(D[0][i], list(contexts.keys())[I[0][i]]) for i in range(len(I[0]))]
        # print("document_similarities", document_similarities)
        if not verbose:
            print("DONE, FAISS")

    return document_similarities
