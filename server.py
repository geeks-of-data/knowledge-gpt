from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from io import BytesIO
from utils_pdf import process_pdf, process_pdf_page
from utils_embedding import compute_doc_embeddings_hf,  compute_doc_embeddings
from utils_completion import answer_query_with_context
from utils_scrape import scrape_content
from utils_subtitles import scrape_youtube
from typing import Optional
from models.search import SearchQuery
from models.scrape import ScrapeQuery
from models.yt_subs import YTSubsQuery
from fastapi import HTTPException


app = FastAPI()

df = None
embeddings = None


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...), extraction_type: Optional[str] = Form("default")):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are allowed"}
    

    pdf_buffer = BytesIO(await file.read())

    global df

    print(extraction_type)

    if extraction_type == "page":
        df = process_pdf_page(pdf_buffer)
    else:
        df = process_pdf(pdf_buffer)

    paragraphs = df.to_dict('records')

    return {"paragraphs": paragraphs}


@app.get("/embedding_calculation/")
async def embedding_calculation():
    global embeddings
    embeddings = compute_doc_embeddings_hf(df, 'tr')
    
    return {"status": "done"}

@app.post("/search/")
async def search(query: SearchQuery):

    target = query.query
    embedding_extractor_type = query.embedding_extractor
    model_lang = query.model_lang

    answer  = answer_query_with_context(target, df, embeddings, embedding_type=embedding_extractor_type, model_lang=model_lang)
    return {"answer": answer}


@app.post("/all_in_one/")
async def all_in_one(query: str = Form(""), extraction_type: str = Form("page"), embedding_extractor: str = Form("hf"), model_lang:str =Form("en") ,file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are allowed"}
    
    print(query)

    pdf_buffer = BytesIO(await file.read())

    global df
    if df is None:
        
        if extraction_type == "page":
            df = process_pdf_page(pdf_buffer)
        else:
            df = process_pdf(pdf_buffer)

    global embeddings
    if embeddings is None:
        if embedding_extractor == "hf":
            embeddings = compute_doc_embeddings_hf(df, model_lang)
        else:
            embeddings = compute_doc_embeddings(df)


    target = query
    answer = ""
    if embedding_extractor == "hf":
        answer  = answer_query_with_context(target, df, embeddings, embedding_type="hf", model_lang=model_lang)
    else:
        answer  = answer_query_with_context(target, df, embeddings, embedding_type="openai", model_lang=model_lang)
    return {"answer": answer}

@app.post("/scrape/")
async def scrape_website(url: ScrapeQuery):
    """
    Endpoint that takes a URL as input and returns the scraped content.
    """
    if not url:
        raise HTTPException(status_code=400, detail="URL is missing")
    
    embedding_extractor = url.embedding_extractor
    model_lang = url.model_lang
    
    global df

    df = scrape_content(url.url)

    global embeddings
    if embeddings is None:
        if embedding_extractor == "hf":
            embeddings = compute_doc_embeddings_hf(df, model_lang)
        else:
            embeddings = compute_doc_embeddings(df)

    target = url.query
    answer = ""

    if embedding_extractor == "hf":
        answer  = answer_query_with_context(target, df, embeddings, embedding_type="hf", model_lang=model_lang)
    else:
        answer  = answer_query_with_context(target, df, embeddings, embedding_type="openai", model_lang=model_lang)

    return {"answer": answer}
    


@app.post("/youtube_subtitles/")
async def youtube_subtitles(query: YTSubsQuery):
    """
    Endpoint that takes a YouTube video ID as input and returns the captions
    as a pandas DataFrame with the key "caption".
    """
    if not query.video_id:
        raise HTTPException(status_code=400, detail="Video ID is missing")
    
    global df

    df = scrape_youtube(query.video_id)

    global embeddings

    if embeddings is None:
        if query.embedding_extractor == "hf":
            embeddings = compute_doc_embeddings_hf(df, query.model_lang)
        else:
            embeddings = compute_doc_embeddings(df)
    
    target = query.query
    answer = ""

    if query.embedding_extractor == "hf":
        answer  = answer_query_with_context(target, df, embeddings, embedding_type="hf", model_lang=query.model_lang)
    else:
        answer  = answer_query_with_context(target, df, embeddings, embedding_type="openai", model_lang=query.model_lang)

    return {"answer": answer}