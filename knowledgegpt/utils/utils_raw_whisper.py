import re
import openai

def transcribe_audio(audio_path):

    paragraphs = []
    sentences_per_paragraph = 6
    
    with open(audio_path, "rb") as f:
        
        transcript = openai.Audio.transcribe("whisper-1", f)
        text = transcript["text"]

        # Split the text into sentences
        sentences = re.findall(r"(.*?\.)(?=\s|$)", text, re.DOTALL)
        sentences = [s.strip() for s in sentences]
        sentences = [s for s in sentences if len(s) > 0]

        # Group the sentences into paragraphs
        num_sentences = len(sentences)
        for i in range(0, num_sentences, sentences_per_paragraph):
            paragraph = ' '.join(sentences[i:i+sentences_per_paragraph])
            paragraphs.append(paragraph)

        return paragraphs