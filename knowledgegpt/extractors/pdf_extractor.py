from ..utils.utils_pdf import process_pdf, process_pdf_page
from ..utils.utils_embedding import compute_doc_embeddings, compute_doc_embeddings_hf
from ..utils.utils_completion import answer_query_with_context
from io import BytesIO
from typing import Optional, Tuple, List


def pdf_scrape(query: str, pdf_file_path: str,mongo_client, max_tokens, extraction_type: str = "page", embedding_extractor: str = "hf", model_lang: str = "en", to_save: bool = False, ) -> Tuple[str, Optional[str], List[str]]:
    
    """
    Extracts paragraphs from a PDF file and computes embeddings for each paragraph, then answers a query using the embeddings.
    :param query: Query to answer
    :param pdf_file_path: Path to the PDF file
    :param extraction_type: Type of extraction to use. Options are "page" and "paragraph"
    :param embedding_extractor: Extractor to use for computing embeddings. Options are "hf" and "openai"
    :param model_lang: Language of the model to use for computing embeddings. Options are "en" and "tr"
    :param to_save: Whether to save the embeddings to MongoDB
    :return: Answer to the query and the prompt and messages
    """

    print("Processing PDF file...")

    with open(pdf_file_path, "rb") as f:
        pdf_file = BytesIO(f.read())
    
    if pdf_file.getvalue()[:4] != b'%PDF':
        raise ValueError("Only PDF files are allowed")
    
    pdf_df = None
    pdf_embeddings = None

    print("Extracting paragraphs...")
    
    if pdf_df is None:
        if extraction_type == "page":
            pdf_df = process_pdf_page(pdf_file)
        else:
            pdf_df = process_pdf(pdf_file)

    print("Computing embeddings...")

    if pdf_embeddings is None:
        if embedding_extractor == "hf":
            pdf_embeddings = compute_doc_embeddings_hf(pdf_df, model_lang)
        else:
            pdf_embeddings = compute_doc_embeddings(pdf_df)

    print("Answering query...")

    target = query
    answer = ""
    if embedding_extractor == "hf":
        answer, prompt, messages = answer_query_with_context(target, pdf_df, pdf_embeddings, embedding_type="hf", model_lang=model_lang, max_tokens=max_tokens)
    else:
        answer, prompt, messages = answer_query_with_context(target, pdf_df, pdf_embeddings, embedding_type="openai", model_lang=model_lang, max_tokens=max_tokens)

    if to_save:
        mongo_client.pair_pdf.insert_one({"query": target, "answer": answer, "prompt": prompt})

    print("Done!")

    return answer, prompt, messages
