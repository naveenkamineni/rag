# 📄 RAG for PTE Academic Q&A

## 📝 Overview
This notebook demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline using a local Ollama LLM and a PDF knowledge source.

---

## ⚙️ Prerequisites
Before running any code cells, ensure you have the following set up:

1. **Python 3.10+** (recommend using a virtual environment)

2. **Required Libraries:** `langchain_community`, `fastembed`, `pypdf`, `chromadb`

3. **Hugging Face Account & Access Token** – for embeddings

4. **Ollama Installed** and required LLM model pulled (`llama3.2` / `mistral`)

5. The target PDF file available locally

---

### Cell 1 - Pre-requisites
📦 **Pre-requisite:** Install required Python packages before running.

```python
%pip install langchain_community
```

### Cell 2 - Pre-requisites
📄 **Pre-requisite:** Ensure your PDF file path is correct and accessible.

```python
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import sys
```

### Cell 3 - Pre-requisites
📄 **Pre-requisite:** Ensure your PDF file path is correct and accessible.

```python
def ingest():
    # Get the doc
    loader = PyPDFLoader(r"C:\Users\kamin\Downloads\PTE_4_Modules_Overview.pdf")
    pages = loader.load_and_split()
    # Split the pages by char
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split {len(pages)} documents into {len(chunks)} chunks.")
    #
    embedding = FastEmbedEmbeddings()
    #Create vector store
    Chroma.from_documents(documents=chunks,  embedding=embedding, persist_directory="./sql_chroma_db")
```

### Cell 4 - Pre-requisites
📦 **Pre-requisite:** Install required Python packages before running.

```python
#%pip install fastembed
#%pip install pypdf
#%pip install chromadb

# only run this once to generate vector store
ingest()
```

### Cell 5 - Pre-requisites
🔑 **Pre-requisite:** You must have a valid Hugging Face access token.

```python
from huggingface_hub import login
access_token_read = "Your_Hugging_face_Token"
access_token_write = "Your_Hugging_face_Token"
login(token = access_token_read)
```

### Cell 6 - Pre-requisites
🤖 **Pre-requisite:** Ollama should be installed and the model (`llama3.2` or chosen) pulled locally.

```python
def rag_chain():
    model = ChatOllama(model="llama3.2")
    #
    prompt = PromptTemplate.from_template(
        """
        "Return the exact phrase from the provided context that answers the question, without adding extra explanation."
        """
        """
        <s> [Instructions] You are a friendly assistant. Answer the question based only on the following context. 
        If you don't know the answer, then reply, No Context availabel for this question {input}. [/Instructions] </s> 
        [Instructions] Question: {input} 
        Context: {context} 
        Answer: [/Instructions]
        """
        
    )
    #Load vector store
    embedding = FastEmbedEmbeddings()
    vector_store = Chroma(persist_directory="./sql_chroma_db", embedding_function=embedding)

    #Create chain
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.5,
        },
    )

    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    #
    return chain
```

### Cell 7 - Pre-requisites
💬 **Pre-requisite:** Make sure ingestion and RAG chain definition cells have been executed.

```python
def ask(query: str):
    #
    chain = rag_chain()
    # invoke chain
    result = chain.invoke({"input": query})
    # print results
    print(result["answer"])
    for doc in result["context"]:
        print("Source: ", doc.metadata["source"])
```

### Cell 8 - Pre-requisites
💬 **Pre-requisite:** Make sure ingestion and RAG chain definition cells have been executed.

```python
ask("What is the main purpose of the Speaking section in the PTE Academic test?")
```
```
The main purpose of the Speaking section in the PTE Academic test is to test oral communication skills and pronunciation. Source: C:\Users\kamin\Downloads\PTE_4_Modules_Overview.pdf Source:
```
### Cell 9 - Pre-requisites
💬 **Pre-requisite:** Make sure ingestion and RAG chain definition cells have been executed.

```python
ask("How many modules in pte Academic?")
```

### Cell 10 - Pre-requisites
💬 **Pre-requisite:** Make sure ingestion and RAG chain definition cells have been executed.

```python
ask("Name any three tasks included in the Speaking module?")
```

### Cell 11 - Pre-requisites
💬 **Pre-requisite:** Make sure ingestion and RAG chain definition cells have been executed.

```python
ask("Which key skill in the Speaking module relates to stress and intonation?")
```

```python

```
