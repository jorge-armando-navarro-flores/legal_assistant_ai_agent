# ============================
# Load Chroma VectorStores as Retrievers
# ============================

import os

# Chroma is the vector database used to store and search document embeddings
from langchain_chroma import Chroma

# OpenAI embeddings model (you must have your OPENAI_API_KEY set)
from langchain_openai import OpenAIEmbeddings


# ============================
# 1. Set Up OpenAI Embeddings
# ============================

# Initialize the embedding function using OpenAI (can be reused across all collections)
embeddings = OpenAIEmbeddings()


# ============================
# 2. Disable Chroma Telemetry (Optional)
# ============================

# This avoids sending anonymous telemetry data when using Chroma locally
os.environ["CHROMA_TELEMETRY"] = "FALSE"


# ============================
# 3. Load Pre-Built VectorStores
# ============================

# Load the vector store for 'laboral' (Labor Law)
laboral_vector_store = Chroma(
    collection_name="laboral",                                  # Name of the Chroma collection
    embedding_function=embeddings,                              # Embedding model used to compare queries
    persist_directory="/home/janf/Projects/legal_assistant_ai_agent/index/laboral/",  # Where vectors are stored on disk
)

# Expose it as a retriever (so it can return relevant documents given a question)
laboral_retriever = laboral_vector_store.as_retriever()


# Load the vector store for 'civil' (Civil Law)
civil_vector_store = Chroma(
    collection_name="civil",
    embedding_function=embeddings,
    persist_directory="/home/janf/Projects/legal_assistant_ai_agent/index/civil/",
)
civil_retriever = civil_vector_store.as_retriever()


# Load the vector store for 'penal' (Criminal Law)
penal_vector_store = Chroma(
    collection_name="penal",
    embedding_function=embeddings,
    persist_directory="/home/janf/Projects/legal_assistant_ai_agent/index/penal/",
)
penal_retriever = penal_vector_store.as_retriever()


# Load the vector store for 'general' (Fallback or uncategorized documents)
general_vector_store = Chroma(
    collection_name="general",
    embedding_function=embeddings,
    persist_directory="/home/janf/Projects/legal_assistant_ai_agent/index/general/",
)
general_retriever = general_vector_store.as_retriever()
