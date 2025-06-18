from langchain_openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI model
llm = OpenAI(model="gpt-3.5-turbo-instruct",temperature=0.5)

# Define a prompt
prompt = "4 line poem on topic playground in funny way"

# Generate a response from the model
result = llm.invoke(prompt)

# Print the response
print(result)
