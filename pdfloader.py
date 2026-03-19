from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

import os

# --- Configuration ---
FILE_PATH = "Health Companion-Health Insurance Plan_GEN617.pdf" 
# Create a dummy file for the example to run if it doesn't exist
if not os.path.exists(FILE_PATH):
    with open(FILE_PATH, "w") as f:
        f.write("LangChain is a framework for developing applications powered by language models. It enables applications to be data-aware and agentic. Data-aware means connecting an LLM to specific external data sources to use in generation. Agentic means allowing an LLM to interact with its environment. This is the first section of the document. RAG stands for Retrieval-Augmented Generation.")

print(f"--- 1. Loading Document: {FILE_PATH} ---")
# Instantiate the PDF Loader
loader = PyPDFLoader(FILE_PATH)

# Load the documents
# This returns a list of Document objects, one for each page of the PDF.
documents = loader.load()

# Display the first document's content and metadata
if documents:
    print(f"Total documents (pages) loaded: {len(documents)}")
    print("\n--- First Document (Page) Content Snippet ---")
    print(documents[0].page_content[:200] + "...") # Show first 200 chars
    print("\n--- First Document Metadata ---")
    print(documents[0].metadata)
else:
    print("No documents were loaded.")



# --- Configuration ---
CHUNK_SIZE = 1000  # Max characters per chunk
CHUNK_OVERLAP = 200 # Number of characters to overlap between adjacent chunks

print("\n--- 2. Splitting Documents ---")
# Instantiate the Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""], # Priority of delimiters
    length_function=len
)

# Split the loaded documents
chunks = text_splitter.split_documents(documents)

# Display the result
print(f"Total chunks created: {len(chunks)}")
if chunks:
    print("\n--- First Chunk Content Snippet ---")
    print(chunks[0].page_content)
    print("\n--- First Chunk Metadata ---")
    print(chunks[0].metadata)



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