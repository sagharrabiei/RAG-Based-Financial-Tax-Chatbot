import streamlit as st
import requests
import os

os.environ["NO_PROXY"] = "127.0.0.1,localhost"  

st.title("Financial Assitant RAG System")

question = st.text_input("Ask your question: ")

if st.button("send"):
    response = requests.post("http://127.0.0.1:8000/ask",json={"question":question})

    st.write(response.json()["answer"])

