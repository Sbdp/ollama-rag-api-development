from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import re

# 1. Initialize the locally running Ollama model
# Replace 'llama3' with the model name you pulled with Ollama
llm = ChatOllama(model="llama3")

# 2. Define a simple prompt template
# Note: The system prompt remains the same to enforce detailed answers
prompt = ChatPromptTemplate.from_messages([
    ("system", ""
    "You are a helpful AI assistant. Don't answer briefly. "
    "Provide a basic definition based on that. Check for follow up questions based on the answer,"
    " ask relevant questions based on the relevant information."),
    ("user", "{input}")
])

# 3. Create a simple chain
chain = prompt | llm

# --- RECURSIVE LOOP START ---
print("--- AI Assistant Loop Started ---")
print("Enter your question in English. Type 'Thank You' or 'NO' to stop the assistant.")

# Start the recursive loop
while True:
    # Get user input
    user_input = input("\nYour Prompt (English): ")
    
    # Normalize input for checking exit condition
    # Use re.IGNORECASE for robust checking
    exit_pattern = re.compile(r'^(thank you|no)$', re.IGNORECASE)
    
    # 5. Check for exit condition
    if exit_pattern.match(user_input.strip()):
        print("\nAssistant: Thank you for using the AI assistant. Goodbye!")
        break
    
    # 6. Invoke the chain with the user's input
    try:
        response = chain.invoke({"input": user_input})
        
        # 7. Print the model's response
        print("-" * 30)
        print("Assistant Response:")
        print(response.content)
        print("-" * 30)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check if your Ollama server is running and the specified model ('llama3') is pulled.")
        break
# --- RECURSIVE LOOP END ---