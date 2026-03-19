import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. App Configuration & UI Setup ---
st.set_page_config(page_title="Llama 3 RAG Chatbot", page_icon="🦙")
st.title("🦙 Llama 3 Local RAG Chatbot")
st.caption("Powered by Ollama and LangChain")

# --- 2. Initialize Components (Cached) ---
@st.cache_resource
def init_rag_components():
    # Load Vector DB
    VECTOR_DB_DIR = "./chroma_db"
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Get top 3 chunks

    # Initialize Local Llama 3 via Ollama
    llm = ChatOllama(model="llama3", temperature=0.2)

    return retriever, llm

retriever, llm = init_rag_components()

# --- 3. Define the RAG Prompt ---
# This tells Llama 3 how to behave and how to use the context.
RAG_PROMPT = """
You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer based on the context, just say that you don't know. 
Keep the answer concise and professional.

Context:
{context}

Question: 
{question}

Answer:
"""
prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT)

# --- 4. Chat History Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm ready! Ask me anything about the document."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. The Chat Interaction ---
if user_query := st.chat_input("What is the insurance coverage?"):
    
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Llama 3 is thinking..."):
            # A. Retrieve context
            docs = retriever.invoke(user_query)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            # B. Build the Chain (Context + Prompt -> LLM -> String)
            chain = (
                {"context": lambda x: context_text, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            # C. Execute and stream/display
            response = chain.invoke(user_query)
            st.markdown(response)
            
            # (Optional) Show sources in an expander
            with st.expander("View Source Chunks"):
                for i, doc in enumerate(docs):
                    st.write(f"**Chunk {i+1}:** {doc.page_content[:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": response})