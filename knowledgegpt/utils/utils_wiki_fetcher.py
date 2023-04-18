def wiki_fetcher(topic):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from knowledgegpt.utils.utils_scrape import scrape_content

    query = 'Baroque Painting'
    endpoint = 'https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={}&utf8='.format(query)

    response = requests.get(endpoint).json()

    search_results = response['query']['search']

    titles = [result['title'] for result in search_results]
    links = ['https://en.wikipedia.org/wiki/{}'.format(result['title'].replace(' ', '_')) for result in search_results]


    df_all = pd.DataFrame()

    for link in links:
      df = scrape_content(link)
      df_all = pd.concat([df_all, df])

    df_all = df_all.reset_index()
    
    df_all.drop(columns=['index'], inplace=True)
    
    return df_all