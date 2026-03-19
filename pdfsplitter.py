from langchain_text_splitters import RecursiveCharacterTextSplitter

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