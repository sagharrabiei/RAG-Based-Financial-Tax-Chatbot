from fastapi import FastAPI
from pydantic import BaseModel
import time



from sentence_transformers import SentenceTransformer
import chromadb
import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

print(os.environ.get("OPENROUTER_API_KEY"))  # باید key رو print کنه نه None


app = FastAPI()

print("loading embedding model...")
t0=time.time()
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print(f"embedding: {time.time()-t0:.2f}s")

print("connecting to chromadb...")
t1=time.time()
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="tax_chunks")
print(f"chromadb: {time.time()-t1:.2f}s")


# generate with openrouter free api



llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")


)



models_to_try = [
    "qwen/qwen3-14b:free",
    "openrouter/auto",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-v3:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-32b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-4b-it:free",

]

#incoming request shape
class QuestionRequst(BaseModel):
    question:str

#response shape
class AnswerResponse(BaseModel):
    answer:str
    

@app.get("/")
def home():
    return {"message": "RAG API is running!"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request:QuestionRequst):

    # Step 1: Embed the question
    question_embedding = embedding_model.encode(request.question).tolist()



    # step2: retrieval
    result_query_chromadb = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )

    context = "\n\n".join(result_query_chromadb["documents"][0])


    messages = [
        {"role": "system", "content": "یک دستیار مالیاتی هستی. بر اساس متن پاسخ بده و سعی کن شفاف توضیحش بدی .جملاتت کامل و با معنا باشد و سوال کاربر را کامل پاسخ دهد. اگر در متن موجود نبود از اطلاعات خودت پاسخ بده ولی مطمئن شو متن فارسی و خوانا است . آمار و اطلاعات را دقیق و کامل بگو. در ضمن ترجمه انگلیسی پاسخ را هم ارسال کن در زیر پاسخ فارسی. اول پاسخ فارسی را ارسال کن . به صورت پاسخ فارسی:       ترجمه: .کاملترین پاسخ را بده "},
        {"role": "user", "content": f"متن:\n{context}\n\nسوال: {request.question}\n\nپاسخ:"}
     ]
    
    #step3: generate: try models in order
    t2 = time.time()
    answer = None
    for model_name in models_to_try:
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
                timeout=15
            )
            answer = response.choices[0].message.content
            break  # if it worked, stop trying
        except Exception as e:
            print(f"مدل {model_name} در دسترس نیست، مدل بعدی را امتحان می‌کنم...")
            continue

    if not answer:
        answer = "در حال حاضر سرویس در دسترس نیست. لطفاً چند دقیقه دیگر امتحان کنید."

    print(f"llm response: {time.time()-t2:.2f}s")

    return AnswerResponse(answer=answer)