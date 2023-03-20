import re
from bs4 import BeautifulSoup
import yt_dlp
import pandas as pd

# chat-gpt wrote this, potentially not the best way to do it :D
def scrape_youtube(url):
    """
    Function that takes a YouTube video URL as input and returns the captions
    as a pandas DataFrame with the key "caption".
    :param url: YouTube video URL
    :return: Pandas DataFrame with the key "caption"
    """

    # Set up youtube-dl options    
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        # 'subtitleslangs': ['en', 'tr'], # Only download English subtitles, this part is disabled for now
        'outtmpl': '%(id)s.%(ext)s' # Use video ID as output filename
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info['id']
        title = info['title']
        filename = f"{video_id}.mp4"
    
        # Download the video and subtitles
        ydl.download([url])

    # Read the subtitles from the file
    with open(f"{video_id}.en.vtt", 'r') as f:
        subtitles = f.read()


    # Convert the subtitles to plain text and remove timestamps (THIS PART IS NOT WORKING)
    soup = BeautifulSoup(subtitles, 'html.parser')
    text = soup.get_text()
    text = re.sub('\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3} ', '', text)

    # Split the text into paragraphs of 3-4 sentences
    paragraph_re = re.compile('\n{2,}')
    sentences_re = re.compile('[\.!?]')
    paragraphs = paragraph_re.split(text)
    paragraphs = [sentences_re.split(p.strip()) for p in paragraphs]
    paragraphs = [[s.strip() for s in sentence_list if s.strip()] for sentence_list in paragraphs]
    paragraphs = [sentence_list for sentence_list in paragraphs if sentence_list]
    
    # Create sequences of at least 30 seconds
    sequence_len = 30
    sequence_sentences = sequence_len * 2  # Assume an average speaking rate of 2 sentences per second
    sequence = []
    sequences = []
    for paragraph in paragraphs:
        if not sequence:
            sequence = paragraph[:sequence_sentences]
        else:
            sequence += paragraph
        if len(sequence) >= sequence_sentences:
            sequences.append({'content': ' '.join(sequence[:sequence_sentences])})
            sequence = sequence[sequence_sentences:]
    
    # Return the sequences in a pandas DataFrame

    df = pd.DataFrame(sequences)
    
    return df