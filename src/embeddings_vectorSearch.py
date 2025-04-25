from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np

def generate_embeddings(documents, model_name='BAAI/bge-small-en-v1.5', batch_size=32):
    """
    Generate embeddings for the context field of documents and add to metadata.

    Args:
        documents (list): List of dictionaries, each with 'context' and 'metadata' fields.
        model_name (str): Name of the SentenceTransformer model (default: BAAI/bge-small-en-v1.5).
        batch_size (int): Batch size for encoding (default: 32).

    Returns:
        list: Updated documents with embeddings added to metadata['context_embedding'].
    """
    model = SentenceTransformer(model_name)
    contexts = [doc['context'] for doc in documents]
    embeddings = model.encode(contexts, batch_size=batch_size, normalize_embeddings=True)
    for doc, embedding in zip(documents, embeddings):
        doc['metadata']['context_embedding'] = embedding.tolist()
    return documents

def create_vector_space(documents, dimension=384):
    """
    Create a FAISS vector space for cosine similarity search.

    Args:
        documents (list): List of document objects with embeddings in metadata['context_embedding'].
        dimension (int): Dimension of the embeddings (default: 384 for BGE-Small).

    Returns:
        faiss.IndexFlatIP: FAISS index for cosine similarity search.
    """
    embeddings = np.array([doc['metadata']['context_embedding'] for doc in documents], dtype=np.float32)
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity (normalized embeddings)
    index.add(embeddings)
    return index

def search_vector_space(query, documents, index, model, k=5):
    """
    Search the vector space for the top-k most similar documents to the query.

    Args:
        query (str): Query string to search for.
        documents (list): List of document objects.
        index (faiss.IndexFlatIP): FAISS index with document embeddings.
        model (SentenceTransformer): Model to encode the query.
        k (int): Number of top results to return (default: 5).

    Returns:
        list: List of (document, similarity_score) tuples for top-k results.
    """
    # Encode the query
    query_embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)

    # Search the index
    distances, indices = index.search(query_embedding, k)

    # Collect results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx >= 0 and idx < len(documents):  # Ensure valid index
            results.append((documents[idx], float(distance)))  # Distance is cosine similarity

    return results

# Main processing
if __name__ == "__main__":
    # Load JSON file
    input_json_path = "scheme_documents.json"  # Replace with your JSON file path
    with open(input_json_path, 'r') as f:
        documents = json.load(f)

    # Generate embeddings
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    documents = generate_embeddings(documents)

    # Create vector space
    vector_space_index = create_vector_space(documents)

    # Save updated documents
    output_json_path = "scheme_documents_with_embeddings.json"
    with open(output_json_path, 'w') as f:
        json.dump(documents, f, indent=2)

    # Save FAISS index
    faiss.write_index(vector_space_index, "scheme_embeddings.faiss")

    # Example search
    query = "Support for small businesses in Assam"
    top_k_results = search_vector_space(query, documents, vector_space_index, model, k=3)

    # Print search results
    print("Top 3 similar documents for query:", query)
    for doc, score in top_k_results:
        print(f"\nSimilarity Score: {score:.4f}")
        print(f"Scheme Name: {doc['metadata']['scheme_name']}")
        print(f"Context: {doc['context'][:200]}...")  # Truncated for brevity