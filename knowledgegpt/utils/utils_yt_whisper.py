from io import BytesIO
import requests
import pydub
import re
import pandas as pd
import openai
import yt_dlp


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
            "preferredquality": "192",
        }]
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    
    # Transcribe audio with OpenAI API
    paragraphs = []
    with open("audio.mp3", "rb") as f:
        # while True:
            # Read a chunk of the file
            # chunk = f.read(25214400) # 25 MB chunk size
            # if not chunk:
            #     # End of file
            #     break
            
            # # Convert chunk to mp3
            # chunks = BytesIO(chunk)
            # chunks.name = "audio.mp3"
            # audio_segment = pydub.AudioSegment.from_file(chunks)
            # mp3_chunk = BytesIO()
            # audio_segment.export(mp3_chunk, format="mp3")
            # mp3_chunk.seek(0)


        transcript = openai.Audio.transcribe("whisper-1", f)
        text = transcript["text"]

        new_paragraphs = re.findall(r"(.*?\.)(?=\s|$)", text, re.DOTALL)
        new_paragraphs = [p.strip() for p in new_paragraphs]
        new_paragraphs = [p for p in new_paragraphs if len(p) > 0]
        
        paragraphs += new_paragraphs
    df = pd.DataFrame({"content": paragraphs})
    return df