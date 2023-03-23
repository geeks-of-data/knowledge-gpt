<!-- Use the context of other files to complete here -->
![knowledgegpt](static_files/logo.png)

# Installation

1. Check Dependencies: ` pip install spacy download en_core_web_sm ffmpeg`

2. Run in terminal:  `pip install knowledgegpt`

## Pypi Link: https://pypi.org/project/knowledgegpt/


# knowledgegpt

***knowledgegpt*** is designed to gather information from various sources, including the internet and local data, which
can be used to create prompts. These prompts can then be utilized by OpenAI's GPT-3 model to generate answers that are
subsequently stored in a database for future reference.

To accomplish this, the text is first transformed into a fixed-size vector using either open source or OpenAI models.
When a query is submitted, the text is also transformed into a vector and compared to the stored knowledge embeddings.
The most relevant information is then selected and used to generate a prompt context.

***knowledgegpt*** supports various information sources including websites, PDFs, PowerPoint files (PPTX), and
documents (Docs). Additionally, it can extract text from YouTube subtitles and audio (using speech-to-text technology)
and use it as a source of information. This allows for a diverse range of information to be gathered and used for
generating prompts and answers.

## How to use

#### Restful API

```uvicorn server:app --reload```

#### Set Your API Key

1. Go to [OpenAI > Account > Api Keys](https://platform.openai.com/account/api-keys)
2. Create new screet key and copy
3. Enter the key to [example_config.py](./examples/example_config.py)

#### How to use the library

```python
# Import the library
from knowledgegpt.extractors.web_scrape_extractor import WebScrapeExtractor

# Import OpenAI and Set the API Key
import openai
from example_config import SECRET_KEY 
openai.api_key = SECRET_KEY

# Define target website
url = "https://en.wikipedia.org/wiki/Bombard_(weapon)"

# Initialize the WebScrapeExtractor
scrape_website = WebScrapeExtractor( url=url, embedding_extractor="hf", model_lang="en")

# Prompt the OpenAI Model
answer, prompt, messages = scrape_website.extract(query="What is a bombard?",max_tokens=300,  to_save=True, mongo_client=db)

# See the answer
print(answer)

# Output: 'A bombard is a type of large cannon used during the 14th to 15th centuries.'

```

Other examples can be found in the [examples](./examples) folder.
But to give a better idea of how to use the library, here is a simple example:

```python
# Basic Usage
basic_extractor = BaseExtractor(df)
answer, prompt, messages = basic_extractor.extract("What is the title of this PDF?", max_tokens=300)
```

```python
# PDF Extraction
pdf_extractor = PDFExtractor( pdf_file_path, extraction_type="page", embedding_extractor="hf", model_lang="en")
answer, prompt, messages = pdf_extractor.extract(query, max_tokens=1500)
```

```python
# PPTX Extraction
ppt_extractor = PowerpointExtractor(file_path=ppt_file_path, embedding_extractor="hf", model_lang="en")
answer, prompt, messages = ppt_extractor.extract( query,max_tokens=500)
```

```python
# DOCX Extraction
docs_extractor = DocsExtractor(file_path="../example.docx", embedding_extractor="hf", model_lang="en", is_turbo=False)
answer, prompt, messages = \
    docs_extractor.extract( query="What is an object detection system?", max_tokens=300)
```

```python
# Extraction from Youtube video (audio)
scrape_yt_audio = YoutubeAudioExtractor(video_id=url, model_lang='tr', embedding_extractor='hf')
answer, prompt, messages = scrape_yt_audio.extract( query=query, max_tokens=1200)

# Extraction from Youtube video (transcript)
scrape_yt_subs = YTSubsExtractor(video_id=url, embedding_extractor='hf', model_lang='en')
answer, prompt, messages = scrape_yt_subs.extract( query=query, max_tokens=1200)
```
## Docker Usage

```bash
docker build -t knowledgegptimage .
docker run -p 8888:8888 knowledgegptimage
```

## How to contribute

0. Open an issue
1. Fork the repo
2. Create a new branch
3. Make your changes
4. Create a pull request

## FEATURES

- [x] Extract knowledge from the internet (i.e. Wikipedia)
- [x] Extract knowledge from local data sources - PDF
- [x] Extract knowledge from local data sources - DOCX
- [x] Extract knowledge from local data sources - PPTX
- [x] Extract knowledge from youtube audio (when caption is not available)
- [x] Extract knowledge from youtube transcripts
- [x] Extract knowledge from whole youtube playlist

## TODO


- [x] FAISS support 
- [ ] Add a vector database (Pinecone, Milvus, Qdrant etc.)
- [x] Add Whisper Model
- [ ] Add Whisper Local Support (not over openai API)
- [ ] Add Whisper for audio longer than 25MB
- [ ] Add a web interface
- [ ] Migrate to Promptify for prompt generation
- [x] Add ChatGPT support
- [ ] Add ChatGPT support with a better infrastructure and planning
- [ ] Increase the number of prompts
- [ ] Increase the number of supported knowledge sources
- [ ] Increase the number of supported languages
- [ ] Increase the number of open source models
- [ ] Advanced web scraping
- [ ] Prompt-Answer storage (the odds are that this will be done in a separate project)
- [ ] Add a better documentation 
- [ ] Add a better logging system
- [ ] Add a better error handling system
- [ ] Add a better testing system
- [ ] Add a better CI/CD system
- [ ] Dockerize the project
- [ ] Add search engine support, such as Google, Bing, etc.
- [ ] Add support for opensource OpenAI alternatives (for answer generation)
- [ ] Evaluating dependecies and removing unnecessary ones
- [ ] Providing prompt flexibility for using with whatever model

( To be extended...)

## System Architecture

<!-- ![System Architecture](static_files/Knowledge-ex.png) -->
(To be updated with a better image)

