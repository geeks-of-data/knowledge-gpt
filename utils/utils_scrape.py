import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import  HTTPException

def scrape_content(url):

    try:
        # Make a GET request to the URL
        response = requests.get(url)
        # Use BeautifulSoup to parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract the text from the HTML
        scraped_content = soup.get_text()
        # Split the text into paragraphs
        paragraphs = [p.strip() for p in scraped_content.split("\n\n") if p.strip()]
        # Create a pandas DataFrame with the scraped content
        df = pd.DataFrame({"content": paragraphs})
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
