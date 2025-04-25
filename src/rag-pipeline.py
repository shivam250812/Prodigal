

# 2. Full implementation with all dependencies
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def retrieve_chunks(query, json_path, faiss_path, model_name='BAAI/bge-small-en-v1.5', k=3):
    """Retrieve relevant document chunks"""
    json_path = "scheme_documents_with_embeddings.json"
    with open(json_path, 'r') as f:
        documents = json.load(f)

    index = faiss.read_index("scheme_embeddings.faiss")
    model = SentenceTransformer(model_name)

    query_embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(query_embedding, k)

    return [
        documents[idx] for idx in indices[0]
        if 0 <= idx < len(documents)
    ]

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

    return f"{system_context}\n\nContext:\n{context}\n\nQuestion: {query}"

def generate_response(system_context, query, json_path, faiss_path):
    """Generate final response using local model"""
    # Create augmented prompt
    prompt = augment_response(system_context, query, json_path, faiss_path)

    # Initialize model
    generator = pipeline(
        'text2text-generation',
        model='google/flan-t5-base',
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16
    )

    # Generate response
    return generator(
        prompt,
        max_length=500,
        do_sample=True,
        temperature=0.7
    )[0]['generated_text']

if __name__ == "__main__":
    # Update these paths to match your actual file locations
    json_path = "scheme_documents_with_embeddings.json"
    faiss_path = "scheme_embeddings.faiss"

    system_context = """You are a helpful assistant providing accurate information about government schemes.
Answer using the context below. Keep responses under 3 sentences.if i ask abou schemes then give me schemes only. if i ask about eligiblity criteria then tell me about eligibility criteria only if i ask about the application process then tell me about application process only if i ask about ministries/departments give me ministries/department only , filter base don tags and context"""

    query = "Scheme for disabled students in kerala"

    response = generate_response(system_context, query, json_path, faiss_path)

    print("Query:", query)
    print("Response:", response)