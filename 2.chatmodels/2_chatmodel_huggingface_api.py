from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Hugging Face Chat model
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation"

)
model = ChatHuggingFace(llm =llm)
# Define a prompt
prompt = "What is the capital of France?"
# Generate a response from the model
result = model.invoke(prompt)
# Print the response
print(result.content)
