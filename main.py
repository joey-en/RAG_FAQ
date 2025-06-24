import os
import streamlit as st
import faiss
import numpy as np

from langchain.document_loaders import PyPDFLoader
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

# ========== PDF LOADING ==========
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

# ========== EMBEDDING + FAISS SETUP ==========

def create_embeddings(text_list):
    try:
        response = client.embeddings.create(model="mistral-embed", inputs=text_list)
        return np.array([r.embedding for r in response.data])
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

def setup_faiss_index(text_chunks):
    embeddings = create_embeddings(text_chunks)
    if embeddings is None:
        st.error("Failed to create embeddings. Cannot initialize FAISS index.")
        return None
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
    st.info("Loading Sakina business proposal and building index...")
    chunks = load_pdf_chunks("data/Sakina BP.pdf")
    index, chunk_texts, embeddings = setup_faiss_index(chunks)
    
    if index:
        st.session_state['faiss_index'] = index
        st.session_state['chunks'] = chunk_texts
        st.session_state['embeddings'] = embeddings
        st.info("Index was created sucessfully")
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
