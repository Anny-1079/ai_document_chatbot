import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from concurrent.futures import ThreadPoolExecutor
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

def embed_texts_parallel(texts, embeddings):
    with ThreadPoolExecutor() as executor:
        vectors = list(executor.map(embeddings.embed_query, texts))
    return vectors

@st.cache_resource
def load_pdf_create_vectorstore(pdf_path, pdf_name):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
    vectorstore_path = f"{pdf_name}_vectorstore"

    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)
        texts = [doc.page_content for doc in docs]

        vectorstore = FAISS.from_texts(texts, embeddings)
        vectorstore.save_local(vectorstore_path)

    return vectorstore

st.title("📄🧠 AI Document Chatbot with Memory")

st.write("Upload your PDF, ask questions, and continue conversation contextually.")

pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file is not None:
    
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    
    vectorstore = load_pdf_create_vectorstore("temp.pdf","temp")
        
    llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = st.session_state.memory

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    user_query = st.text_input("Ask a question about your document:")

    if user_query:
        result = qa({"question": user_query})
        st.write("🤖 Answer:", result["answer"])

        with st.expander("Show conversation history"):
            for i, msg in enumerate(memory.chat_memory.messages):
                st.write(f"{msg.type}: {msg.content}")

