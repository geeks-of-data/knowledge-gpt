<!-- Use the context of other files to complete here -->
# Knowledgebase 

This is a repo for extracting knowledge from the internet or from other local data sources to form-up prompts.

## How to use
uvicorn server:app --reload

## How to contribute
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
- [x] Extract knowledge from youtube captions


## TODO
- [ ] Add a database
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

( To be extended...)

## System Architecture

![System Architecture](./public/Knowledge-ex.png)