import docx
import pandas as pd

def extract_paragraphs(filename):

    """
    Extracts paragraphs from a Word file
    :param filename: Path to the Word file
    :return: Dataframe with paragraphs
    """

    doc = docx.Document(filename)
    paragraphs = []
    for p in doc.paragraphs:
        sentences = p.text.split('. ')
        for i in range(0, len(sentences), 5):
            paragraph = '. '.join(sentences[i:i+5])
            paragraphs.append(paragraph)
    return pd.DataFrame({"content":paragraphs})