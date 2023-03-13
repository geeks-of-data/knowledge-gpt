from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from io import BytesIO
from utils.utils_pdf import process_pdf, process_pdf_page
from utils.utils_embedding import compute_doc_embeddings_hf,  compute_doc_embeddings
from utils.utils_completion import answer_query_with_context
from utils.utils_scrape import scrape_content
from utils.utils_subtitles import scrape_youtube
from utils.utils_powerpoint import process_pptx
from typing import Optional
from models.search import SearchQuery
from models.scrape import ScrapeQuery
from models.yt_subs import YTSubsQuery
from fastapi import HTTPException


app = FastAPI()

df = None
embeddings = None

pdf_df = None
pdf_embeddings = None

pptx_df = None
pptx_embeddings = None

web_df = None
web_embeddings = None

yt_sub_df = None
yt_sub_embeddings = None

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...), extraction_type: Optional[str] = Form("default")):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are allowed"}

    pdf_buffer = BytesIO(await file.read())

    print(extraction_type)

    global pdf_df

    if extraction_type == "page":
        pdf_df = process_pdf_page(pdf_buffer)
    else:
        pdf_df = process_pdf(pdf_buffer)

    paragraphs = pdf_df.to_dict('records')

    return {"paragraphs": paragraphs}


@app.get("/embedding_calculation/")
async def embedding_calculation():

    global pdf_embeddings
    global pdf_df

    pdf_embeddings = compute_doc_embeddings_hf(pdf_df, 'tr')
    
    return {"status": "done"}

@app.post("/search/")
async def search(query: SearchQuery):

    global pdf_df
    global pdf_embeddings

    target = query.query
    embedding_extractor_type = query.embedding_extractor
    model_lang = query.model_lang

    answer  = answer_query_with_context(target, pdf_df, pdf_embeddings, embedding_type=embedding_extractor_type, model_lang=model_lang)
    return {"answer": answer}


@app.post("/all_in_one/")
async def all_in_one(query: str = Form(""), extraction_type: str = Form("page"), embedding_extractor: str = Form("hf"), model_lang:str =Form("en") ,file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are allowed"}
    
    print(query)

    pdf_buffer = BytesIO(await file.read())

    global pdf_df
    global pdf_embeddings

    if pdf_df is None:
        
        if extraction_type == "page":
            pdf_df = process_pdf_page(pdf_buffer)
        else:
            pdf_df = process_pdf(pdf_buffer)

    if pdf_embeddings is None:
        if embedding_extractor == "hf":
            pdf_embeddings = compute_doc_embeddings_hf(pdf_df, model_lang)
        else:
            pdf_embeddings = compute_doc_embeddings(pdf_df)


    target = query
    answer = ""
    if embedding_extractor == "hf":
        answer  = answer_query_with_context(target, pdf_df, pdf_embeddings, embedding_type="hf", model_lang=model_lang)
    else:
        answer  = answer_query_with_context(target, pdf_df, pdf_embeddings, embedding_type="openai", model_lang=model_lang)
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
    
    global web_df
    global web_embeddings

    web_df = scrape_content(url.url)


    if web_embeddings is None:
        if embedding_extractor == "hf":
           web_embeddings = compute_doc_embeddings_hf(web_df, model_lang)
        else:
           web_embeddings = compute_doc_embeddings(web_df)

    target = url.query
    answer = ""

    if embedding_extractor == "hf":
        answer  = answer_query_with_context(target, web_df,web_embeddings, embedding_type="hf", model_lang=model_lang)
    else:
        answer  = answer_query_with_context(target, web_df,web_embeddings, embedding_type="openai", model_lang=model_lang)

    return {"answer": answer}
    


@app.post("/youtube_subtitles/")
async def youtube_subtitles(query: YTSubsQuery):
    """
    Endpoint that takes a YouTube video ID as input and returns the captions
    as a pandas DataFrame with the key "caption".
    """
    if not query.video_id:
        raise HTTPException(status_code=400, detail="Video ID is missing")
    
    global yt_sub_df
    global yt_sub_embeddings

    yt_sub_df = scrape_youtube(query.video_id)


    if yt_sub_embeddings is None:
        if query.embedding_extractor == "hf":
           yt_sub_embeddings = compute_doc_embeddings_hf(yt_sub_df, query.model_lang)
        else:
           yt_sub_embeddings = compute_doc_embeddings(yt_sub_df)
    
    target = query.query
    answer = ""

    if query.embedding_extractor == "hf":
        answer  = answer_query_with_context(target, yt_sub_df,yt_sub_embeddings, embedding_type="hf", model_lang=query.model_lang)
    else:
        answer  = answer_query_with_context(target, yt_sub_df,yt_sub_embeddings, embedding_type="openai", model_lang=query.model_lang)

    return {"answer": answer}





@app.post("/powerpoint_scrape/")
async def powerpoint_scrape(query: str = Form(""),  embedding_extractor: str = Form("hf"), model_lang:str =Form("en") ,file: UploadFile = File(...)):

    if file.content_type != "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        return {"error": "Only Powerpoint files are allowed"}
    
    global pptx_df
    global pptx_embeddings

    pptx_buffer = BytesIO(await file.read())
    pptx_df = process_pptx(pptx_buffer)


    if pptx_embeddings is None:
        if embedding_extractor == "hf":
           pptx_embeddings = compute_doc_embeddings_hf(pptx_df, model_lang)
        else:
           pptx_embeddings = compute_doc_embeddings(pptx_df)

    target = query
    answer = ""

    if embedding_extractor == "hf":
        answer  = answer_query_with_context(target, pptx_df,pptx_embeddings, embedding_type="hf", model_lang=model_lang)
    else:
        answer  = answer_query_with_context(target, pptx_df,pptx_embeddings, embedding_type="openai", model_lang=model_lang)

    return {"answer": answer}
