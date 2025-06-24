import os
import streamlit as st
import faiss
import numpy as np

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from mistralai import Mistral, UserMessage

# ========== CONFIG ==========

# Set the API key as an environment variable
# os.environ["MISTRAL_API_KEY"] = "P5Ya1Is7YS4AM2dVkBU0KrV9Bz0BU0KU"

# Retrieve the API key using os.getenv() 
from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.getenv("MISTRAL_API_KEY", "MISTRAL_API_KEY not found")
client = Mistral(api_key=api_key)

PDF_FOLDER_PATH = "./data"
INDEX_PATH = "./saved_index_chunks/faiss.index"
CHUNK_PATH = "./saved_index_chunks/chunks.pkl"

# ========== CONTENT LOADING ==========
def load_pdf_chunks(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]

def load_txt_chunks(pdf_path):
    loader = TextLoader(pdf_path, encoding="utf-8")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]

def load_pdf_chunks_from_folder(folder_path):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            all_chunks.extend(load_pdf_chunks(os.path.join(folder_path, filename)))
        if filename.endswith(".txt"):
            all_chunks.extend(load_txt_chunks(os.path.join(folder_path, filename)))
    return all_chunks

# -------

import pickle

def save_faiss_index(index, path=INDEX_PATH):
    faiss.write_index(index, path)

def load_faiss_index(path=INDEX_PATH):
    return faiss.read_index(path)

def save_chunks(chunks, path=CHUNK_PATH):
    with open(path, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks(path=CHUNK_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)

# ========== EMBEDDING + FAISS SETUP ==========
import time
def create_embeddings(text_list, batch_size=30, delay=2.0): # Added batch size because of API limits
    all_embeddings = []
    list_size = len(text_list)
    progress_bar = st.progress(0, text=f"Embedding {list_size} chunks in progress...")
    
    try:
        for i in range(0, list_size, batch_size):
            batch = text_list[i:i + batch_size]
            response = client.embeddings.create(model="mistral-embed", inputs=batch)
            embeddings = [r.embedding for r in response.data]
            all_embeddings.extend(embeddings)

             # Delay to avoid rate limit
            time.sleep(delay)
            percent_done = min((i + batch_size), list_size) / list_size
            progress_bar.progress(percent_done, text=f"Embedding: {i} of {list_size} chunks ({int(percent_done * 100)}%) ")

        progress_bar.empty() # Remove bar
        return np.array(all_embeddings)
    
    except Exception as e:
        st.error(f"Error in batch {i}–{i+batch_size}: {text_list[i]}... \n\n ------ \n\n {e}")
        return None
    
    
def setup_faiss_index(text_chunks):
    embeddings = create_embeddings(text_chunks)
    if embeddings is None:
        st.error("Failed to create embeddings. Cannot initialize FAISS index.")
        return None, None, None  # <–– return a tuple regardless of error to avoid more errors
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks, embeddings

# ========== CONTEXT RETRIEVAL ==========

def fetch_relevant_chunks(query, index, chunks, num_chunks=3):
    query_embedding = create_embeddings([query])
    if query_embedding is None:
        return []
    _, indices = index.search(query_embedding, num_chunks)
    return [chunks[i] for i in indices[0]]

# ========== LLM RESPONSE GENERATION ==========

def ask_mistral(context_chunks, query):
    context = "\n".join(context_chunks)
    prompt = (
        f"""
        
        You are a Sakina's supportive mental health assistant. 
        Use the information below to answer the user's concern in a helpful and professional tone. 

        For every interaction, ask me any question that will help you respond more personally and effectively.
        When explaining mental health concepts, start with a high-level overview to empathize, 
        then break the situation or concept into smaller digestable blocks; use analogies if they help.

        Make sure to greet the user and use emojis when appropriate.\n\n
        ---
        Context:\n{context}\n\n
        ---
        User Query: {query}\n
        ---
        Supportive Response:
        """
    )
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[UserMessage(content=prompt)]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response from Mistral: {e}")
        return "Sorry, something went wrong. Please try again."

# ========== STREAMLIT UI ==========

st.set_page_config(page_title="Mental Health Chatbot Sakina Ai", page_icon="🧠")
st.title("🧠 Mental Health Support Chatbot Sakina AI")
st.markdown("_This tool provides general mental health support and is **not** a substitute for professional help. If you're in crisis, please contact a professional or emergency service._")

# Initialize chunks and FAISS index once
if 'chunks' not in st.session_state:
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNK_PATH):
        st.info("Loading cached index and chunks...")
        st.session_state['faiss_index'] = load_faiss_index(INDEX_PATH)
        st.session_state['chunks'] = load_chunks(CHUNK_PATH)
        st.success("Cached data loaded.")
    else: # Read folder for the first time
        st.info("Fetching relevant clinical information and building database...")
        chunks = load_pdf_chunks_from_folder(PDF_FOLDER_PATH)
        index, chunk_texts, embeddings = setup_faiss_index(chunks)
        
        if index:
            st.session_state['faiss_index'] = index
            st.session_state['chunks'] = chunk_texts
            st.session_state['embeddings'] = embeddings
            st.info("Index was created sucessfully")

            save_faiss_index(index, INDEX_PATH)
            save_chunks(chunks, CHUNK_PATH)
        else:
            st.error("Failed to build FAISS index.")

# User input
user_query = st.text_input("How are you feeling today, or what would you like support with?")

if st.button("Start Chat"):
    if user_query.strip() and 'faiss_index' in st.session_state:
        context_chunks = fetch_relevant_chunks(user_query, st.session_state['faiss_index'], st.session_state['chunks'])
        if context_chunks:
            answer = ask_mistral(context_chunks, user_query)
            st.markdown(f"**SakinaAI Agent:**\n\n{answer}")
        else:
            st.warning("Sorry, I couldn't find relevant context. Please try again.")
    else:
        st.warning("Please enter something you'd like help with.")
