import os
import torch

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

#reading file
with open("inta_texts_cleaned.txt", "r", encoding="utf-8") as file:
    text = file.read()

print(len(text))
#chunking and create a list of chunks
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

chunks = chunk_text(text)
# print(len(chunks_list))
# print("***************************************************")
# print(chunks_list[0])
# print("****************************************************")
# print(chunks_list[1])

#embeddings we want to convert each chunk of text to a list of numbers(vectors) that represents its meaning similar mining = similar number
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="tax_chunks")


if collection.count() == 0:
    print("converting chuncks to vectors ...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    print(f"number of chunks: {embeddings.shape[0]}")
    print(f"size of each vector: {embeddings.shape[1]}")
    print(embeddings[:5])

    #everything should be added as list
    collection.add(
        documents=chunks, #original text of each chunk
        embeddings=embeddings.tolist(), #the 384 numbers for each chunks. convert numpy array to list for chromadb to accepts
        ids=[str(i) for i in range(len(chunks))] #we have 131 chunks

    )

    print(f"{collection.count()} chunks have been saved to chromadb database")


else:
    print(f"{collection.count()} chunks are already saved to chromadb database!")

# Retrieval (core of rag) the user asks a question we convert it to vector of numbers using the same model then ask
#chromadb  which chunks are closest in meaning to this question vector




from transformers import AutoTokenizer, AutoModelForCausalLM


# load tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained("./models/qwen")
#by default uses cpu
#model_llm = AutoModelForCausalLM.from_pretrained("./models/qwen")

#using gpu
# model_llm = AutoModelForCausalLM.from_pretrained(
#     "./models/qwen",
#     dtype=torch.float16,  # uses half the memory
#     device_map="cuda"
# )

print("سامانه آماده است. برای خروج 'خروج' بنویسید")

while True:
    question = input("\nسوال: ").strip()

    if question == "خروج":
        print("خداحافظ!")
        break

    if not question:
        continue

    # retrieval
    question_embedding = model.encode(question).tolist()
    result_query_chromadb = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )

    context = "\n\n".join(result_query_chromadb["documents"][0])


    messages = [
        {"role": "system", "content": "یک دستیار مالیاتی هستی. بر اساس متن پاسخ بده و سعی کن شفاف توضیحش بدی .جملاتت کامل و با معنا باشد و سوال کاربر را کامل پاسخ دهد. اگر در متن موجود نبود بنویس اطلاعات در دیتاست موجود نیست و از اطلاعات خودت پاسخ بده ولی مطمئن شو متن فارسی و خوانا است . آمار و اطلاعات را دقیق و کامل بگو. در ضمن ترجمه انگلیسی پاسخ را هم ارسال کن در زیر پاسخ فارسی. اول پاسخ فارسی را ارسال کن . به صورت پاسخ فارسی:       ترجمه: .کاملترین پاسخ را بده "},
        {"role": "user", "content": f"متن:\n{context}\n\nسوال: {question}\n\nپاسخ:"}
     ]

    # generate with openrouter free api
    from dotenv import load_dotenv

    load_dotenv()
    from openai import OpenAI

    llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")


    )

    models_to_try = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "openrouter/auto",
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen3-32b:free",
        "qwen/qwen3-14b:free",
        "deepseek/deepseek-v3:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-3-27b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemma-3-12b-it:free",
        "google/gemma-3-4b-it:free",

    ]

    answer = None
    for model_name in models_to_try:
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            answer = response.choices[0].message.content
            break  # if it worked, stop trying
        except Exception as e:
            print(f"مدل {model_name} در دسترس نیست، مدل بعدی را امتحان می‌کنم...")
            continue

    if not answer:
        answer = "در حال حاضر سرویس در دسترس نیست. لطفاً چند دقیقه دیگر امتحان کنید."

    print(f"\nپاسخ: {answer}")

    # generate with local model from hugging face

    # input_text = tokenizer.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # #using cpu
    # inputs = tokenizer(input_text, return_tensors="pt")
    #
    # #for using gpu
    # #inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    #
    # output = model_llm.generate(
    #     **inputs,
    #     max_new_tokens=150,
    #     temperature=0.3,
    #     do_sample=True,
    #     pad_token_id=tokenizer.eos_token_id,
    #
    # )
    #
    # answer = tokenizer.decode(
    #     output[0][inputs["input_ids"].shape[1]:],
    #     skip_special_tokens=True
    # )
    #
    #print(f"\nپاسخ: {answer}")
