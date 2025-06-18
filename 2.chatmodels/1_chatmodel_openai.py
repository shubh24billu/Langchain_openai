from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI Chat model
model = ChatOpenAI(model="gpt-4", temperature=0.7, max_tokens=15)

result = model.invoke("What is the capital of France?")


# Print the result
print(result.content)