<!-- Use the context of other files to complete here -->
![knowledgegpt](./public/logo.png)

# knowledgegpt 

***knowledgegpt*** is designed to gather information from various sources, including the internet and local data, which can be used to create prompts. These prompts can then be utilized by OpenAI's GPT-3 model to generate answers that are subsequently stored in a database for future reference.

To accomplish this, the text is first transformed into a fixed-size vector using either open source or OpenAI models. When a query is submitted, the text is also transformed into a vector and compared to the stored knowledge embeddings. The most relevant information is then selected and used to generate a prompt context.

***knowledgegpt*** supports various information sources including websites, PDFs, PowerPoint files (PPTX), and documents (Docs). Additionally, it can extract text from YouTube subtitles and audio (using speech-to-text technology) and use it as a source of information. This allows for a diverse range of information to be gathered and used for generating prompts and answers.

## How to use

#### Restful API

```uvicorn server:app --reload```

#### How to install the library

```pip install knowledgegpt```
or
```
git clone https://github.com/geeks-of-data/knowledge-gpt.git
pip install .
```

#### How to use the library

```
# Import the library
from knowledgegpt.extractors.web_scrape_extractor import scrape_website

# Import OpenAI and Set the API Key
import openai
from config import SECRET_KEY
openai.api_key = SECRET_KEY


# If you want to use mongodb to store the data
from config import MONGO_URI
from pymongo import MongoClient

client  = MongoClient(MONGO_URI)
db = client.openai_test

# Define target website
url = "https://en.wikipedia.org/wiki/Bombard_(weapon)"

# Prompt the OpenAI Model
answer, prompt, messages = scrape_website(db, url, embedding_extractor="hf", model_lang="en", max_tokens=200, query="What is a bombard?", to_save=True)

# See the answer
print(answer)

# Output: 'A bombard is a type of large cannon used during the 14th to 15th centuries.'

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
- [x] Library implementation (partially done, initial release)


## TODO
- [x] Add a database (partially done)
- [ ] Add a vector database
- [x] Add Whisper Model
- [ ] Add Whisper for audio longer than 25MB
- [ ] Add a web interface
- [ ] Migrate to Promptify
- [x] Add ChatGPT support (only in docs endpoint and experimental)
- [ ] Add ChatGPT support with a better infrastructure and planning
- [ ] Increase the number of prompts
- [ ] Increase the number of supported knowledge sources
- [ ] Increase the number of supported languages
- [ ] Increase the number of open source models
- [ ] Dockerize the project
- [ ] Advanced web scraping
- [ ] Prompt-Answer storage
- [ ] Add a better documentation
- [ ] Check library functions to see if they are working properly
- [ ] Add a better logging system
- [ ] Add a better error handling system
- [ ] Add a better testing system

( To be extended...)

## System Architecture

![System Architecture](./public/Knowledge-ex.png)