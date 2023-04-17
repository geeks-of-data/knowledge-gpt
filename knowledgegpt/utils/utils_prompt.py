# https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb sourced from here
from knowledgegpt.utils.utils_distance import order_document_sections_by_query_similarity
import pandas as pd

import tiktoken

SEPARATOR = "\n* "
ENCODING = "gpt2"

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

relevancy_template = '''
You duty is to check if the question given and the context part given are relevant to each other. If they are relevant, please write "yes" or "y" or "1" or "true" or "t". If they are not relevant, please write "no" or "n" or "0" or "false" or "f". If you are not sure, please write "unsure" or "u" or "2" or "maybe" or "m".
You don't have to be super strict, a basic relevancy check is enough we are trying to hunt down stuff like references to other documents, or other stuff that is not relevant to the question.
Question: {question}
Context: {context}
Answer:
'''


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame, embedding_type: str = "hf",
                     verbose=False, model_lang: str = "en", max_tokens=1000, index_type="basic", prompt_template=None, strict_context=False) -> str:
    """
    Construct the prompt to be used for completion.
    :param question: The question to answer.
    :param context_embeddings: The embeddings of the document sections.
    :param df: The dataframe containing the document sections.
    :param embedding_type: The type of embedding used. Can be "hf" or "tf".
    :param model_lang: The language of the model. Can be "en" or "tr".
    :param max_tokens: The maximum number of tokens to be used in turbo model.
    :return: The prompt to be used for completion.
    """
    MAX_SECTION_LEN = max_tokens

    most_relevant_document_sections = order_document_sections_by_query_similarity(
        query=question,
        contexts=context_embeddings,
        embedding_type=embedding_type,
        model_lang=model_lang,
        verbose=verbose,
        index_type=index_type
    )

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    if strict_context:
        print("STRICT MODE IS ON, THIS IS GOING TO TAKE A WHILE AND IS AN EXPERIMENTAL FEATURE")
    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[section_index]
        
        if strict_context:
            if len(document_section.content) < 10:
                continue
            
            import openai
            from knowledgegpt.utils.utils_completion import model_types
            
            prompt = relevancy_template.format(question=question, context=document_section.content)
            
            response = openai.Completion.create(
                prompt = prompt,
                ** model_types["davinci"]
            )
            
            if response["choices"][0]["text"].strip(" \n").lower() in ["no", "n", "0", "false", "f"]:
                continue
        
        document_tokens = len(encoding.encode(document_section.content))
        chosen_sections_len += document_tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    if not verbose:
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))

    if prompt_template is None:
        if model_lang == "tr":
            header = """Cümleyi doğru bir şekilde cevaplayın ve cevap metin içinde yoksa "bilmiyorum" diyin.\n\nMetin:\n"""
        else:
            header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

        return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"
    else:
        return prompt_template.format(question=question, sections="".join(chosen_sections))
