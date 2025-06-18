from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

# Initialize the OpenAI Embeddings model
embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

# Sample documents
documents = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome.",
    "Paris is known for its art, fashion, and culture.",
    "Berlin is the capital city of Germany, known for its history and modern landmarks."
]   

query = "What is the capital of France?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Calculate cosine similarity between the query embedding and document embeddings
print(cosine_similarity([query_embedding], doc_embeddings))

# Find the most similar document
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("similarty score:", score)

