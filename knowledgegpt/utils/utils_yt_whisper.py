from io import BytesIO
import requests
import pydub
import re
import pandas as pd
import openai
import yt_dlp
from knowledgegpt.utils.utils_raw_whisper import transcribe_audio

# currently not working for files > 25MB
# TODO: split audio into chunks of 25MB and transcribe each chunk

def transcribe_youtube_audio(url):
    """
    Function that takes a YouTube video URL as input and returns the captions
    as a pandas DataFrame with the key "caption".
    :param url: YouTube video URL
    :return: Pandas DataFrame with the key "caption"
    """
    options = {
        "format": "bestaudio/best",
        "outtmpl": "audio.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            # "preferredquality": "192", # can be activated later on to get better quality
            "preferredquality": "64"
        }]
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])

    paragraphs = transcribe_audio("audio.mp3")
    df = pd.DataFrame({"content": paragraphs})

    return df