from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI Embeddings model
embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32
                             
                             )
documents = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome."
]

# Generate embeddings for the documents
result = embedding.embed_documents(documents)
# Print the embedding vectors   
print(str(result))
