import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import HTTPException

def scrape_content(url):
    import cloudscraper
    """
    Extracts paragraphs from a webpage
    :param url: URL of the webpage
    :return: Dataframe with paragraphs
    """
    scraper = cloudscraper.create_scraper(browser='chrome')  

    try:
        response = scraper.get(url)
        
        if response.status_code != 200:
            response = requests.get(url, headers={"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"})
        
        soup = BeautifulSoup(response.content, 'html.parser')
        scraped_content = soup.get_text()
        paragraphs = [p.strip() for p in scraped_content.split("\n\n") if p.strip()]
        df = pd.DataFrame({"content": paragraphs})
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
