# RAG powered DocBot with LangChain and ChromaDB

An end-to-end **Retrieval-Augmented Generation (RAG)** system that turns your PDFs into a **conversational DocBot**.  
It uses **LangChain** for orchestration, **Hugging Face transformer embeddings** for vectorization, **ChromaDB** for vector storage & retrieval (with FAISS as a drop-in alternative), and a **Groq-hosted Gemma 2 9B Instruct** LLM for fast, grounded responses.

---

## 🔍 What this project does

- 🧠 **Ingests unstructured PDFs** using `UnstructuredPDFLoader` and converts them into clean, chunked texts.
- ✂️ **Splits text into retrieval-friendly chunks** using `CharacterTextSplitter` (configurable `chunk_size` / `chunk_overlap`).
- 🔡 **Embeds chunks** with `HuggingFaceEmbeddings` (Transformer sentence embeddings).
- 🗃️ **Indexes into ChromaDB** (persistent on-disk vector DB) for **vector similarity search**.
- 🤝 **Composes a QA chain** (`load_qa_chain(chain_type="stuff")`) to ground the LLM on retrieved context.
- ⚡ **Generates answers** via **Groq** `ChatGroq` with **Gemma 2 9B IT**, keeping responses *concise* and *context-aware*.

> **Note:** The reference notebook also includes a FAISS example. This README defaults to **ChromaDB**, with a snippet to switch between stores.

---

## 🧩 Architecture

```
PDFs  ──► UnstructuredPDFLoader ──► Text Splitter ──► HF Embeddings ──► ChromaDB
                                                    ▲                        │
                                                    │                        ▼
                                                Query (user) ──► Retriever ──► Context
                                                                                 │
                                                                                 ▼
                                                                        ChatGroq (Gemma 2 9B IT)
                                                                                 │
                                                                                 ▼
                                                                    Grounded Answer (RAG Output)
```

---

## 🛠️ Tech Stack

- **LangChain** (`langchain`, `langchain_community`, `langchain_core`)
- **Vector DB**: **ChromaDB** (persistent) — optional **FAISS** (in-memory)
- **Embeddings**: `HuggingFaceEmbeddings` (e.g., `all-MiniLM-L6-v2`)
- **Document Loading**: `UnstructuredPDFLoader` (`unstructured[local-inference]`)
- **LLM**: **Groq** `ChatGroq` with **Gemma 2 9B IT`
- **Utilities**: `tiktoken`, `Cython`, `numpy`, `pandas`, `matplotlib` (as needed)

---

## ⚙️ Setup

1) **Install dependencies**
```bash
pip install -q langchain langchain_core langchain_community langchain_groq   chromadb sentence-transformers Cython tiktoken unstructured[local-inference]
```

2) **Set environment variables**
```bash
export GROQ_API_KEY="your_groq_api_key"     # Windows (Powershell): $env:GROQ_API_KEY="..."
```

3) **Put your PDFs** in a `data/` folder (e.g., `data/your_docs.pdf`).

---

## 🧪 Minimal Example (ChromaDB)

```python
# 1) Load documents
from langchain.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader("data/your_docs.pdf")
documents = loader.load()

# 2) Split into chunks
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 3) Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()  # defaults to all-MiniLM-L6-v2

# 4) Vector store: ChromaDB (persistent)
from langchain_community.vectorstores import Chroma
vectordb = Chroma(collection_name="docbot", embedding_function=embeddings, persist_directory="chroma_store")
vectordb.add_documents(docs)
vectordb.persist()

# 5) Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# 6) LLM (Groq + Gemma 2 9B IT)
from langchain_groq import ChatGroq
llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="gemma2-9b-it")

# 7) QA chain (stuff)
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff")

# 8) Ask a question
query = "Give me a 5-bullet summary of key topics in this document."
docs = retriever.get_relevant_documents(query)
answer = chain.run(input_documents=docs, question=query)
print(answer)
```

---

## 🔄 FAISS Alternative (Drop-in)

```python
from langchain.vectorstores import FAISS
faiss_db = FAISS.from_documents(docs, embeddings)
docs = faiss_db.similarity_search("List the Generative AI technologies mentioned", k=4)
answer = chain.run(input_documents=docs, question="List the Generative AI technologies mentioned")
print(answer)
```

---

## ✅ Tips for Quality & Performance

- Tune `chunk_size` / `chunk_overlap` based on document density (e.g., 800–1200 / 100–200).
- Experiment with higher-quality embeddings (e.g., `sentence-transformers/all-mpnet-base-v2`).
- Adjust retriever `k` for broader/narrower context windows.
- Use `chain_type="map_reduce"` for very long answers or large corpora.
- Consider response-length control via system prompts for concise outputs.

---

## 🔒 Security

- Keep your `GROQ_API_KEY` private (use env vars or secret managers).
- Do not commit `chroma_store/` unless you intend to share the vectors.

---

## 📣 Acknowledgements

- **LangChain** community docs and examples
- **Groq** for ultra-low-latency inference
- **Hugging Face** for open-source embeddings
- **ChromaDB** for developer-friendly, persistent vector storage


## Thank You
