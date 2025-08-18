## RAG-powered DocBot with LangChain and ChromaDB

- A Retrieval-Augmented Generation (RAG) based document assistant built with LangChain, ChromaDB, and state-of-the-art LLMs (Gemma/GROQ/HuggingFace). This project enables intelligent document ingestion, semantic search, and context-aware answering from PDFs or unstructured text.

## Features

* Document Ingestion – Load PDFs and unstructured text using Unstructured loaders.

* Chunking & Embedding – Process documents into vector embeddings with Sentence-Transformers.

* Vector Store – Store and query embeddings using ChromaDB or FAISS.

* Semantic Retrieval – Retrieve the most relevant chunks for a query.

* LLM-Powered Responses – Generate accurate answers with LangChain orchestration and Gemma/GROQ models.

* High-Performance – Leverages GROQ API for ultra-fast inference.

## Tech Stack

- LangChain – Agentic AI framework for RAG pipelines

- ChromaDB / FAISS – Vector database for embeddings

- Sentence-Transformers – Embedding model for semantic similarity

- Gemma (Hugging Face) – Open-source LLM

- GROQ API – High-speed LLM inference

- Python – Core programming language

## Project Workflow

* Document Loading – PDFs and unstructured data loaded via LangChain loaders

* Preprocessing – Text chunking and cleaning

* Embeddings Generation – Using sentence-transformers

* Vector Storage – ChromaDB stores embeddings for retrieval

* Query Execution – Retrieve top-k chunks relevant to user queries

* LLM Integration – Gemma/GROQ generates context-aware answers

* Response Delivery – Final answer presented in natural language

## Thank You
