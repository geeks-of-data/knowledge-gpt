import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import  HTTPException

def scrape_content(url):
    """
    Extracts paragraphs from a webpage
    :param url: URL of the webpage
    :return: Dataframe with paragraphs
    """

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        scraped_content = soup.get_text()
        paragraphs = [p.strip() for p in scraped_content.split("\n\n") if p.strip()]
        df = pd.DataFrame({"content": paragraphs})
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
