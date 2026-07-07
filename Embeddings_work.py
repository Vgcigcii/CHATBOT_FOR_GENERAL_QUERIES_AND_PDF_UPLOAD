from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_file_chunks(input_list):
    embedded_contents = model.encode(input_list)
    return embedded_contents
def embed_user_query(user_query):
    embedded_user_query = model.encode([user_query])
    return embedded_user_query
def calculate_cosine_similarity_then_retrieval(embeddings1,embeddings2):
    similarity  = cosine_similarity(embeddings1, embeddings2).flatten()
    top_indices = np.argsort(similarity)[::-1]
    return top_indices
