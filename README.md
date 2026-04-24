# Persian Tax Assistant - RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about Iranian tax regulations in Persian (Farsi). The system combines semantic search with large language models to provide accurate, context-aware responses with English translations.

##  Overview

This project implements a production-ready RAG pipeline that:
- Processes Persian tax documentation into searchable chunks
- Uses semantic embeddings to find relevant information
- Generates comprehensive answers using state-of-the-art LLMs
- Provides bilingual responses (Persian primary, English translation)

Perfect for tax professionals, businesses, and individuals navigating Iranian tax regulations.

###  Dataset Creation

Since no ready-made Persian tax dataset was available, I **built a custom web crawler** to automatically scrape and compile tax regulations from [tax.gov.ir](https://tax.gov.ir) - the official Iranian tax authority website. The crawler:
- Navigates through the official tax documentation pages
- Extracts and cleans Persian text content
- Handles pagination and nested document structures
- Produces the `inta_texts_cleaned.txt` dataset

This demonstrates **end-to-end data engineering** - from raw web data to a production-ready knowledge base.

##  Features

- **Custom Dataset**: Built a web crawler to scrape tax.gov.ir and create the dataset from scratch
- **Semantic Search**: Uses multilingual sentence transformers to understand Persian queries
- **Vector Database**: ChromaDB for efficient similarity search and persistent storage
- **Smart Chunking**: Overlapping text chunks to preserve context across boundaries
- **Multiple LLM Fallback**: Automatically tries different free models if one fails
- **Flexible Deployment**: Switch between cloud APIs and local models without code changes
- **Bilingual Output**: Answers in Persian with English translations
- **Offline Capable**: Can run with locally cached models

##  Tech Stack

- **Python 3.8+**
- **Web Scraping**: Custom crawler for tax.gov.ir data collection
- **ChromaDB**: Vector database for semantic search
- **Sentence Transformers**: `paraphrase-multilingual-MiniLM-L12-v2` for embeddings
- **OpenRouter API**: Access to multiple free LLM providers
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library (supports local models too)

##  Prerequisites

```bash
pip install torch chromadb sentence-transformers transformers python-dotenv openai
```

##  Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/sagharrabiei/persian-tax-assistant.git
cd persian-tax-assistant




### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Get your free API key from [OpenRouter](https://openrouter.ai/).

### 4. Run the System

```bash
python main.py
```

The first run will:
1. Process and chunk your tax documentation
2. Generate embeddings for each chunk
3. Store everything in ChromaDB
4. Start the interactive Q&A session

Subsequent runs will use the cached database instantly.

##  Usage Example

```
سامانه آماده است. برای خروج 'خروج' بنویسید

سوال: مالیات بر ارزش افزوده چیست؟

پاسخ: مالیات بر ارزش افزوده نوعی مالیات غیرمستقیم است که...
Translation: Value Added Tax (VAT) is a type of indirect tax that...
```

Type `خروج` to exit the system.

##  Architecture

### 0. **Data Collection** (Preprocessing)
- Custom web crawler scrapes tax.gov.ir
- Extracts Persian text from government website
- Cleans and formats raw HTML content
- Produces structured text dataset

### 1. **Document Processing**
- Reads tax documentation from text file
- Splits into 500-character chunks with 50-character overlap
- Preserves context across chunk boundaries

### 2. **Embedding Generation**
- Uses `paraphrase-multilingual-MiniLM-L12-v2` model
- Converts text to 384-dimensional vectors
- Supports Persian, English, and 50+ languages

### 3. **Vector Storage**
- Persistent ChromaDB database (`./chroma_db`)
- Stores documents, embeddings, and IDs
- Efficient similarity search

### 4. **Retrieval & Generation**
- Converts user question to embedding
- Retrieves top 5 most relevant chunks
- Combines chunks into context
- Generates answer using LLM with system prompt

### 5. **LLM Fallback Chain**
The system tries models in order until one succeeds:
1. NVIDIA Nemotron 120B
2. OpenRouter Auto-select
3. Meta Llama 3.3 70B
4. Qwen 3 32B
5. Qwen 3 14B
6. DeepSeek V3
7. Google Gemma models
8. Mistral Small
9. And more...

##  Project Structure

```
.
├── main.py                    # Main RAG application
├── inta_texts_cleaned.txt     # Scraped and cleaned tax documentation
├── .env                       # Environment variables
├── chroma_db/                 # Vector database (auto-generated)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```



Use local LLM inference (uncommented in code) for complete privacy and zero data transmission.

##  Use Cases

- **Tax Professionals**: Quick lookup of tax regulations
- **Businesses**: Understanding compliance requirements
- **Developers**: Building tax-aware applications
- **Researchers**: Analyzing tax policy documentation
- **Students**: Learning about Iranian tax system

##  Skills Demonstrated

This project showcases:

- **Data Engineering**: Built a web crawler to scrape and clean data from government websites
- **NLP & Embeddings**: Implemented semantic search using multilingual transformers
- **Vector Databases**: Designed efficient retrieval system with ChromaDB
- **LLM Integration**: Multi-provider fallback system with OpenRouter API
- **Local Model Deployment**: Configured both CPU and GPU inference with Hugging Face
- **Production Architecture**: Modular code supporting multiple deployment scenarios
- **Bilingual Systems**: Generated structured Persian-English responses
- **Problem Solving**: Created a dataset from scratch when none existed publicly

