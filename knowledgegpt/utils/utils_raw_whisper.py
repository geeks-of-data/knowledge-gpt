import re
import openai

def transcribe_audio(audio_path):

    paragraphs = []
    
    with open(audio_path, "rb") as f:
        
        transcript = openai.Audio.transcribe("whisper-1", f)
        text = transcript["text"]

        new_paragraphs = re.findall(r"(.*?\.)(?=\s|$)", text, re.DOTALL)
        new_paragraphs = [p.strip() for p in new_paragraphs]
        new_paragraphs = [p for p in new_paragraphs if len(p) > 0]
        
        paragraphs += new_paragraphs

        return paragraphs