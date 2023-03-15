import PyPDF2
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')

def process_pdf(pdf_file):
    """
    Extracts paragraphs from a PDF file
    :param pdf_file: PDF file
    :return: Dataframe with paragraphs
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    num_pages =  len(pdf_reader.pages)

    paragraphs = []

    for i in range(num_pages):
        page = pdf_reader.pages[i]

        text = page.extract_text()

        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        paragraph = ''
        for sentence in sentences:
            if len(paragraph) + len(sentence) > 500:
                paragraphs.append(paragraph)
                paragraph = sentence
            else:
                paragraph += ' ' + sentence
        paragraphs.append(paragraph)

    df = pd.DataFrame({'content': paragraphs})
    return df

def process_pdf_page(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    df_cook = pd.DataFrame(columns=['page_number', 'content'])

    for page_num in range( len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()

        df_cook = df_cook.append({'page_number': page_num+1, 'content': text}, ignore_index=True)

    return df_cook