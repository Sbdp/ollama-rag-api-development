from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" # A high-performing, small, open-source model

print("\n--- 3. Embedding and Indexing ---")

# 3a. Initialize the Embedding Model
# We set device to 'cpu' for non-GPU environments.
embeddings = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print(f"Using Embedding Model: {EMBEDDING_MODEL_NAME}")

# 3b. Create the Vector Store (Index) from chunks
# Chroma.from_documents handles generating embeddings and storing them.
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DB_DIR
)
print(f"Vector Store created and saved to: {VECTOR_DB_DIR}")

# Optional: Verify the retrieval capability
retriever = vectorstore.as_retriever()
query = "What is the purpose of LangChain?"
retrieved_docs = retriever.invoke(query)

print("\n--- Verification: Retrieval Test ---")
print(f"Query: '{query}'")
print(f"Top 1 retrieved chunk content:\n{retrieved_docs[0].page_content}")

# Clean up / close the persistence (important for Chroma)
vectorstore.persist()