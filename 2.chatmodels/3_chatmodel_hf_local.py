from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


# Initialize the HuggingFace model
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.1,
        max_new_tokens=100, # Adjust max_new_tokens as needed
    )
)

# Create a ChatHuggingFace instance
model = ChatHuggingFace(llm=llm)    

# Define a prompt
result = model.invoke("What is the capital of France?")

# Print the response
print(result.content)  # Access the content of the response