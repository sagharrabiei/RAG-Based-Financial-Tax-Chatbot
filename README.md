# RAG-Based Financial & Tax Chatbot

## Overview
This project is a **Retrieval-Augmented Generation (RAG) chatbot** designed to answer financial and tax-related questions using real-world data from Iranian tax regulations.

The chatbot processes user queries and generates context-aware responses based on a custom-built dataset collected from official sources.

---

## Features
- Interactive chatbot (CLI-based)
- Context-aware question answering (RAG approach)
- Uses real-world financial & tax data
- 🇮🇷 Supports **Persian (Farsi)** language output
- Fast responses using API-based LLMs

---

## How It Works
1. A **Python web crawler** collects tax-related data from public sources.
2. The data is cleaned and stored in a text file.
3. The chatbot:
   - Takes user input
   - Injects relevant context into the prompt
   - Sends it to a language model
4. The model generates a **context-aware response**

---

## Technologies Used
- Python
- OpenAI-compatible API
- Prompt Engineering
- Web Scraping
- RAG (Retrieval-Augmented Generation)

---

## Project Structure
.
├── main.ipynb
├── inta_texts_cleaned.txt
├── requirements.txt
└── README.md


## Language Note
The chatbot responses are in Persian (Farsi) because the dataset is based on Iranian tax regulations.
This demonstrates the ability to work with non-English datasets and multilingual NLP systems.


## Example
##  نمونه پرسش و پاسخ (Example Q&A)

###  سوال (Question)
**مالیات بر ارزش افزوده چقدر است؟**  
Translation: *What is the value-added tax (VAT) rate?*

###  پاسخ (Answer)
طبق بند (خ) تبصره (۱) ماده واحده قانون بودجه سال ۱۴۰۴، نرخ مالیات بر ارزش افزوده برای اشخاص مشمول این قانون از **۹ درصد به ۱۰ درصد** افزایش یافته است. بنابراین، مالیات بر ارزش افزوده معادل **۱۰٪** می‌باشد.  

Translation: *According to clause (Kh) of Note (1) in the 1404 budget law, the VAT rate for applicable taxpayers has increased from **9% to 10%**. Therefore, the VAT rate is **10%**.*


Data Source
Public data collected from:
Iranian National Tax Administration (tax.gov.ir)

Note:

All data used in this project is publicly available
Extracted via a custom-built web crawler
