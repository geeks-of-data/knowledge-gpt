from fastapi import FastAPI, File, UploadFile, Form
from io import BytesIO
from utils_pdf import process_pdf, process_pdf_page
from utils_embedding import compute_doc_embeddings_hf, get_hf_embeddings
from utils_distance import order_document_sections_by_query_similarity
from utils_prompt import construct_prompt
from utils_completion import answer_query_with_context

from pydantic import BaseModel

class SearchQuery(BaseModel):
    query: str


app = FastAPI()



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are allowed"}
    

    pdf_buffer = BytesIO(await file.read())

    global df

    df = process_pdf_page(pdf_buffer)

    paragraphs = df.to_dict('records')

    return {"paragraphs": paragraphs}


@app.get("/embedding_calculation/")
async def embedding_calculation():
    global embeddings
    embeddings = compute_doc_embeddings_hf(df)
    
    return {"status": "done"}

@app.post("/search/")
async def search(query: SearchQuery):

    target = query.query

    answer  = answer_query_with_context(target, df, embeddings)
    return {"answer": answer}


# @app.post("/all_in_one/")
# async def all_in_one(query: SearchQuery = Form(...), file: UploadFile = File(...)):
#     if file.content_type != "application/pdf":
#         return {"error": "Only PDF files are allowed"}
    

#     pdf_buffer = BytesIO(await file.read())

#     global df

#     df = process_pdf_page(pdf_buffer)

#     # paragraphs = df.to_dict('records')

#     global embeddings
#     embeddings = compute_doc_embeddings_hf(df)

#     target = query

#     answer  = answer_query_with_context(target, df, embeddings)
#     return {"answer": answer}