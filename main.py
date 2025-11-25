# DEBUG ENABLED main.py — commit this temporarily, redeploy, get logs, then remove debug block

import sys, traceback, logging, os
import streamlit as st

# Setup logging to show in Streamlit logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- DEBUG: Attempt imports and capture failures with full tracebacks ---
import_ok = {}
import_tracebacks = {}

def try_import(name, import_fn):
    try:
        obj = import_fn()
        import_ok[name] = True
        return obj
    except Exception as e:
        import_ok[name] = False
        import_tracebacks[name] = traceback.format_exc()
        logger.exception("Import failed: %s", name)
        return None

# Test core imports used by your app
langchain_mod = try_import("langchain", lambda: __import__("langchain"))
# old-style text splitter import (your code)
text_splitter_mod = try_import("langchain.text_splitter",
                               lambda: __import__("langchain.text_splitter", fromlist=["*"]))
# new-style text splitters package
text_splitters_pkg = try_import("langchain_text_splitters",
                                lambda: __import__("langchain_text_splitters"))
# langchain_community
langchain_community_mod = try_import("langchain_community", lambda: __import__("langchain_community"))
# langchain_groq
langchain_groq_mod = try_import("langchain_groq", lambda: __import__("langchain_groq"))
# langchain_huggingface
langchain_hf_mod = try_import("langchain_huggingface", lambda: __import__("langchain_huggingface"))
# faiss
faiss_mod = try_import("faiss", lambda: __import__("faiss"))
# sentence-transformers (embedding dependencies)
st_mod = try_import("sentence_transformers", lambda: __import__("sentence_transformers"))

# Show quick diagnostic info in sidebar
st.sidebar.header("⚠️ Debug info (temporary)")
st.sidebar.markdown(f"Python: {sys.version.split()[0]}")
st.sidebar.markdown(f"Working dir: {os.getcwd()}")
st.sidebar.markdown("**Import checks:**")
for k in ["langchain", "langchain.text_splitter", "langchain_text_splitters",
          "langchain_community", "langchain_groq", "langchain_huggingface",
          "faiss", "sentence_transformers"]:
    st.sidebar.markdown(f"- {k}: {'✅' if import_ok.get(k) else '❌'}")

# Print the tracebacks (if any) into logs (and make them visible in UI below)
if any(not v for v in import_ok.values()):
    st.sidebar.markdown("**Tracebacks (click to expand)**")
    for name, ok in import_ok.items():
        if not ok:
            st.sidebar.text(f"--- {name} traceback ---")
            st.sidebar.text(import_tracebacks.get(name, "No traceback captured"))

# If any critical import failed, render a clear error in UI and stop app early
critical_imports = ["langchain", "langchain.text_splitter", "langchain_community"]
if any(not import_ok.get(c, False) for c in critical_imports):
    st.title("❌ Startup import error — debug info shown in sidebar")
    st.error("One or more required modules failed to import. See the DEBUG INFO in the sidebar and the Streamlit logs.")
    # Also write tracebacks to main area for convenience
    st.subheader("Tracebacks")
    for name, ok in import_ok.items():
        if not ok:
            st.text(f"--- {name} traceback ---")
            st.text(import_tracebacks.get(name, "No traceback captured"))
    # Stop further execution so the app doesn't crash with generic "Oh no"
    st.stop()

# --- End DEBUG import checks. After you fix, remove the above debug block. ---

# If imports succeeded, continue with original app code (paste below)
try:
    # your original imports (wrapped to use already-imported modules)
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.chains import ConversationalRetrievalChain
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_groq import ChatGroq
    from concurrent.futures import ThreadPoolExecutor
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from dotenv import load_dotenv
except Exception as e:
    logger.exception("Error importing runtime modules after debug checks: %s", e)
    st.error("Runtime imports failed. See logs.")
    st.exception(e)
    st.stop()

# --- your existing app logic follows unchanged ---
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
    try:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())

        vectorstore = load_pdf_create_vectorstore("temp.pdf","temp")
            
        llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")

        if "store" not in st.session_state:
            st.session_state.store = {}

        def get_session_history(session_id: str):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(),
        )

        qa_with_history = RunnableWithMessageHistory(
            qa,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

        user_query = st.text_input("Ask a question about your document:")

        if user_query:
            result = qa_with_history.invoke(
                {"question": user_query},
                config={"configurable": {"session_id": "user1"}},
            )
            st.write("🤖 Answer:", result["answer"])

            with st.expander("Show conversation history"):
                history = get_session_history("user1")
                for i, msg in enumerate(history.messages):
                    st.write(f"{msg.type}: {msg.content}")
    except Exception as e:
        logger.exception("Runtime error in PDF handling / QA flow: %s", e)
        st.error("Runtime error — see logs for full traceback")
        st.exception(e)



# import os
# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from langchain_community.chat_message_histories import ChatMessageHistory   # <── fixed here
# from langchain_groq import ChatGroq
# from concurrent.futures import ThreadPoolExecutor
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from dotenv import load_dotenv

# load_dotenv()

# groq_api_key = os.getenv("GROQ_API_KEY")

# def embed_texts_parallel(texts, embeddings):
#     with ThreadPoolExecutor() as executor:
#         vectors = list(executor.map(embeddings.embed_query, texts))
#     return vectors

# @st.cache_resource
# def load_pdf_create_vectorstore(pdf_path, pdf_name):
#     embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
#     vectorstore_path = f"{pdf_name}_vectorstore"

#     if os.path.exists(vectorstore_path):
#         vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
#     else:
#         loader = PyPDFLoader(pdf_path)
#         pages = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
#         docs = text_splitter.split_documents(pages)
#         texts = [doc.page_content for doc in docs]

#         vectorstore = FAISS.from_texts(texts, embeddings)
#         vectorstore.save_local(vectorstore_path)

#     return vectorstore

# st.title("📄🧠 AI Document Chatbot with Memory")

# st.write("Upload your PDF, ask questions, and continue conversation contextually.")

# pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if pdf_file is not None:
    
#     with open("temp.pdf", "wb") as f:
#         f.write(pdf_file.read())
    
#     vectorstore = load_pdf_create_vectorstore("temp.pdf","temp")
        
#     llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")

#     # --- 🔄 Replace ConversationBufferMemory with RunnableWithMessageHistory ---
#     if "store" not in st.session_state:
#         st.session_state.store = {}

#     def get_session_history(session_id: str):
#         if session_id not in st.session_state.store:
#             st.session_state.store[session_id] = ChatMessageHistory()
#         return st.session_state.store[session_id]

#     qa = ConversationalRetrievalChain.from_llm(
#         llm,
#         retriever=vectorstore.as_retriever(),
#     )

#     qa_with_history = RunnableWithMessageHistory(
#         qa,
#         get_session_history,
#         input_messages_key="question",
#         history_messages_key="chat_history",
#     )

#     user_query = st.text_input("Ask a question about your document:")

#     if user_query:
#         # Use invoke with session_id
#         result = qa_with_history.invoke(
#             {"question": user_query},
#             config={"configurable": {"session_id": "user1"}},
#         )

#         st.write("🤖 Answer:", result["answer"])

#         with st.expander("Show conversation history"):
#             history = get_session_history("user1")
#             for i, msg in enumerate(history.messages):
#                 st.write(f"{msg.type}: {msg.content}")




# # import os
# # import streamlit as st
# # from langchain_community.document_loaders import PyPDFLoader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_community.vectorstores import FAISS   
# # from langchain.chains import ConversationalRetrievalChain
# # from langchain.memory import ConversationBufferMemory , ChatMessageHistory
# # from langchain_groq import ChatGroq
# # from concurrent.futures import ThreadPoolExecutor
# # from langchain_huggingface import HuggingFaceEmbeddings  
# # from langchain_core.runnables.history import RunnableWithMessageHistory
# # from dotenv import load_dotenv

# # load_dotenv()

# # groq_api_key = os.getenv("GROQ_API_KEY")

# # def embed_texts_parallel(texts, embeddings):
# #     with ThreadPoolExecutor() as executor:
# #         vectors = list(executor.map(embeddings.embed_query, texts))
# #     return vectors

# # @st.cache_resource
# # def load_pdf_create_vectorstore(pdf_path, pdf_name):
# #     embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
# #     vectorstore_path = f"{pdf_name}_vectorstore"

# #     if os.path.exists(vectorstore_path):
# #         vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
# #     else:
# #         loader = PyPDFLoader(pdf_path)
# #         pages = loader.load()
# #         text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
# #         docs = text_splitter.split_documents(pages)
# #         texts = [doc.page_content for doc in docs]

# #         vectorstore = FAISS.from_texts(texts, embeddings)
# #         vectorstore.save_local(vectorstore_path)

# #     return vectorstore

# # st.title("📄🧠 AI Document Chatbot with Memory")

# # st.write("Upload your PDF, ask questions, and continue conversation contextually.")

# # pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# # if pdf_file is not None:
    
# #     with open("temp.pdf", "wb") as f:
# #         f.write(pdf_file.read())
    
# #     vectorstore = load_pdf_create_vectorstore("temp.pdf","temp")
        
# #     llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")

# #     if "memory" not in st.session_state:
# #         st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# #     memory = st.session_state.memory

# #     qa = ConversationalRetrievalChain.from_llm(
# #         llm,
# #         retriever=vectorstore.as_retriever(),
# #         memory=memory
# #     )

# #     user_query = st.text_input("Ask a question about your document:")

# #     if user_query:
# #         result = qa.invoke({"question": user_query})   

# #         st.write("🤖 Answer:", result["answer"])

# #         with st.expander("Show conversation history"):
# #             for i, msg in enumerate(memory.chat_memory.messages):
# #                 st.write(f"{msg.type}: {msg.content}")
