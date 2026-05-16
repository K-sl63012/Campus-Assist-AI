# =========================================
# STEP 1 — LOAD ENV VARIABLES
# =========================================

from dotenv import load_dotenv
load_dotenv()

import os



# =========================================
# STEP 2 — IMPORT LIBRARIES
# =========================================

# Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI


# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector Database
from langchain_community.vectorstores import FAISS

# RetrievalQA

from langchain_classic.chains import RetrievalQA

# Prompt Template
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader



# =========================================
# STEP 3 — LOAD MULTIPLE PDF DOCUMENTS
# =========================================

print("\nLoading PDF documents...")

import os

from langchain_community.document_loaders import PyPDFLoader

documents = []

folder_path = "data"

for file in os.listdir(folder_path):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(folder_path, file)

        print(f"Loading: {file}")

        loader = PyPDFLoader(pdf_path)

        documents.extend(loader.load())

print(f"\nLoaded {len(documents)} pages from all PDFs")


# =========================================
# STEP 4 — CHUNK DOCUMENTS
# =========================================

print("\nSplitting document into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

print(f"Created {len(docs)} chunks")


# =========================================
# STEP 5 — CREATE EMBEDDINGS
# =========================================

print("\nCreating embeddings...")

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# =========================================
# STEP 6 — CREATE VECTOR DATABASE
# =========================================

print("\nCreating FAISS vector database...")

db = FAISS.from_documents(
    docs,
    embedding
)

print("FAISS vector database created successfully")


# =========================================
# STEP 7 — CREATE RETRIEVER
# =========================================

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# =========================================
# STEP 8 — LOAD GEMINI MODEL
# =========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    google_api_key='AIzaSyAmRdniJlnQVEf8DT6wC7RY17yFFU1bLF0'
)



# =========================================
# STEP 9 — CREATE CUSTOM PROMPT
# =========================================

prompt_template = """

You are CampusAssist AI.

Answer questions only from the provided context.

If exact answer is not available,
try to provide the closest relevant answer from context.

If nothing relevant exists, say:
"I don't have enough information."

Give short and clear answers.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# =========================================
# STEP 10 — CREATE RetrievalQA CHAIN
# =========================================

qa_chain = RetrievalQA.from_chain_type(

    llm=llm,

    chain_type="stuff",

    retriever=retriever,

    return_source_documents=True,

    chain_type_kwargs={
        "prompt": PROMPT
    }
)


# =========================================
# STEP 11 — START CHAT LOOP
# =========================================

print("\nRAG RetrievalQA Chatbot Ready!")
print("Type 'exit' to stop.\n")

while True:

    question = input("Enter your question: ")

    if question.lower() == "exit":
        break

    # =====================================
    # STEP 12 — GET RESPONSE
    # =====================================

    result = qa_chain.invoke(
        {"query": question}
    )

    # =====================================
    # STEP 13 — PRINT ANSWER
    # =====================================

    print("\n==============================")
    print("ANSWER:\n")

    print(result["result"])

    print("==============================\n")


    