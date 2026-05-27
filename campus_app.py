import streamlit as st

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Campus Assist AI",
    page_icon="🎓",
    layout="wide"
)

# =========================================
# CUSTOM COLORFUL UI
# =========================================

page_bg = """
<style>

[data-testid="stAppViewContainer"]{
background: linear-gradient(
135deg,
#0f2027,
#203a43,
#2c5364,
#6a11cb,
#2575fc
);
background-size: 400% 400%;
animation: gradient 15s ease infinite;
color: white;
}

@keyframes gradient {
0% {
background-position: 0% 50%;
}
50% {
background-position: 100% 50%;
}
100% {
background-position: 0% 50%;
}
}

[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}

.main-title{
font-size: 50px;
font-weight: bold;
text-align: center;
color: white;
padding-top: 10px;
text-shadow: 2px 2px 20px cyan;
}

.sub-title{
font-size: 32px;
font-weight: bold;
text-align: center;
margin-bottom: 30px;
animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {

0%{
color: #ffffff;
text-shadow:
0 0 5px #00ffff,
0 0 10px #00ffff,
0 0 20px #00ffff;
transform: scale(1);
}

100%{
color: #ffccff;
text-shadow:
0 0 10px #ff00ff,
0 0 20px #ff00ff,
0 0 40px #ff00ff;
transform: scale(1.05);
}

}

.question-box{
background-color: rgba(255,255,255,0.1);
padding: 20px;
border-radius: 20px;
backdrop-filter: blur(10px);
}

.answer-box{
background-color: rgba(0,0,0,0.3);
padding: 20px;
border-radius: 20px;
margin-top: 20px;
font-size: 20px;
color: white;
box-shadow: 0px 0px 15px cyan;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================

st.markdown(
    """
    <div class="main-title">
        🎓 Santhiram Engineering College
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
        Campus Assist AI 🤖
    </div>
    """,
    unsafe_allow_html=True
)


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


@st.cache_resource
def load_rag():

    documents = []

    folder_path = "data"

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(folder_path, file)

            print(f"Loading: {file}")

            loader = PyPDFLoader(pdf_path)

            documents.extend(loader.load())

    print(f"\nLoaded {len(documents)} pages from all PDFs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    print(f"Created {len(docs)} chunks")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(
        docs,
        embedding
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

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

    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

    qa_chain = RetrievalQA.from_chain_type(

        llm=llm,

        chain_type="stuff",

        retriever=retriever,

        return_source_documents=True,

        chain_type_kwargs={
            "prompt": PROMPT
        }
    )

    return qa_chain

qa_chain = load_rag()
# =========================================
# STREAMLIT QUESTION INPUT UI
# =========================================

st.markdown("<br>", unsafe_allow_html=True)

question = st.text_input(
    "Ask your question:",
    placeholder="Example: Who is the principal?"
)

# =========================================
# ASK BUTTON
# =========================================

if st.button("Ask CampusAssist AI"):

    if question:

        with st.spinner("Searching answer..."):

            result = qa_chain.invoke(
                {"query": question}
            )

            answer = result["result"]

        st.markdown(
            f"""
            <div class="answer-box">
                <h3>Answer:</h3>
                <p>{answer}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
