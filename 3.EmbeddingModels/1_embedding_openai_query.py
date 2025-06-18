from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv  

load_dotenv

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

result = embeddings.embed_query("What is the capital of France?")
print(result)  # Print the embedding vector
