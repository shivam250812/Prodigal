import streamlit as st
import json
import numpy as np
import faiss
import torch
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import warnings

# Suppress torch.classes warnings
os.environ["STREAMLIT_WATCH_EXCLUDES"] = "torch._classes"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="streamlit")

# Set page configuration
st.set_page_config(
    page_title="Government Schemes Assistant",
    page_icon="🏛️",
    layout="wide"
)

# Add minimal CSS for styling
st.markdown("""
<style>
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }
    .response-container {
        margin-top: 20px;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .source-container {
        margin-top: 20px;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    .source-title {
        font-weight: bold;
        color: #1976D2;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading to improve performance
@st.cache_resource
def load_model(model_name='BAAI/bge-small-en-v1.5'):
    return SentenceTransformer(model_name)

@st.cache_resource
def load_documents_and_index(json_path, faiss_path):
    try:
        # Load documents
        with open(json_path, 'r') as f:
            documents = json.load(f)
        
        # Load FAISS index
        index = faiss.read_index(faiss_path)
        
        return documents, index
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
        return None, None
    except Exception as e:
        st.error(f"Error loading documents or index: {str(e)}")
        return None, None

@st.cache_resource
def load_generator():
    return pipeline(
        'text2text-generation',
        model='google/flan-t5-base',
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

def retrieve_chunks(query, json_path, faiss_path, model_name='BAAI/bge-small-en-v1.5', k=3):
    """Retrieve relevant document chunks"""
    documents, index = load_documents_and_index(json_path, faiss_path)
    if documents is None or index is None:
        return []
    
    model = load_model(model_name)
    
    # Encode the query
    query_embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)
    
    # Search the index
    distances, indices = index.search(query_embedding, k)
    
    # Collect results
    results = [
        documents[idx] for idx in indices[0]
        if 0 <= idx < len(documents)
    ]
    
    return results

def augment_response(system_context, query, json_path, faiss_path, max_context_length=512):
    """Augment query with retrieved context"""
    # Retrieve documents
    documents = retrieve_chunks(
        query=query,
        json_path=json_path,
        faiss_path=faiss_path
    )
    
    # Build context string
    context = "\n".join([d["context"] for d in documents])[:max_context_length]
    
    return f"{system_context}\n\nContext:\n{context}\n\nQuestion: {query}", documents

def generate_response(system_context, query, json_path, faiss_path):
    """Generate final response using local model"""
    # Create augmented prompt
    prompt, source_documents = augment_response(system_context, query, json_path, faiss_path)
    
    # Get generator
    generator = load_generator()
    
    # Generate response
    response = generator(
        prompt,
        max_length=500,
        do_sample=True,
        temperature=0.7
    )[0]['generated_text']
    
    return response, source_documents

# Main app
st.title("Government Schemes Assistant")

# Simple query input
query = st.text_input("What government scheme are you looking for?", 
                      placeholder="E.g., Support for small businesses in Assam")

# File paths - adjust these to your actual file locations
json_path = r"scheme_documents_with_embeddings.json"
faiss_path = r"scheme_embeddings.faiss"

# System prompt for the model
system_context = """You are a helpful assistant providing accurate information about government schemes.
Answer using the context below. Keep responses under 3 sentences. If asked about schemes, give schemes only.
If asked about eligibility criteria, tell about eligibility criteria only. If asked about application process,
tell about application process only. If asked about ministries/departments, give ministries/departments only.
Filter based on tags and context."""

# Process query
if query:
    with st.spinner("Finding relevant information..."):
        try:
            # Generate response using the full RAG pipeline
            response, source_documents = generate_response(system_context, query, json_path, faiss_path)
            
            # Display the generated response
            st.markdown("### Answer")
            st.markdown(f'<div class="response-container">{response}</div>', unsafe_allow_html=True)
            
            # Display source information (collapsible)
            with st.expander("View Sources"):
                if source_documents:
                    for doc in source_documents:
                        st.markdown(f"""
                        <div class="source-container">
                            <div class="source-title">{doc['metadata']['scheme_name']}</div>
                            <div>{doc['context']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No source documents found.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")