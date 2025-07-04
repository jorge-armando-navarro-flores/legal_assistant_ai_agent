# ============================
# PDF Indexing Script with LangChain + Chroma
# ============================

import os

# PDF loader that converts PDFs into LangChain documents
from langchain_community.document_loaders import PyPDFLoader

# Embedding model from OpenAI (requires OPENAI_API_KEY in env)
from langchain_openai import OpenAIEmbeddings

# Used to split long text into chunks for better embedding and retrieval
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store to save and search embedded documents
from langchain_chroma import Chroma

# Optional: Turn off Chroma's telemetry data collection
os.environ["CHROMA_TELEMETRY"] = "FALSE"


# ============================
# 1. Walk through the docs/ folder recursively
# ============================

# This loop traverses all subdirectories and files under the /docs folder
for dirpath, dirnames, filenames in os.walk(
    "/home/janf/Projects/legal_assistant_ai_agent/docs"
):
    # Get the name of the current folder (used as collection/category name)
    parent_dir = dirpath.split("/")[-1]

    for filename in filenames:
        # Full path to the current PDF file
        pdf_path = os.path.join(dirpath, filename)

        # ============================
        # 2. Define Indexing Metadata
        # ============================

        # Directory where the vector index will be stored
        persist_directory = f"index/{parent_dir}/"

        # Collection name for Chroma (usually same as category/folder name)
        collection_name = parent_dir

        # ============================
        # 3. Load and Process PDF
        # ============================

        # Load the PDF using PyPDFLoader (each page becomes a document)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # ============================
        # 4. Create Embeddings
        # ============================

        # Initialize the embedding model (uses your OpenAI API key)
        embeddings = OpenAIEmbeddings()

        # ============================
        # 5. Set up Chroma Vector Store
        # ============================

        # Create or load a vector database (per collection/category)
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory  # Save to disk
        )

        # ============================
        # 6. Split Documents into Chunks
        # ============================

        # Large documents are split into overlapping text chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,       # Each chunk is up to 1000 characters
            chunk_overlap=200      # Chunks overlap by 200 characters
        )

        # Apply the splitter to the loaded PDF documents
        all_splits = text_splitter.split_documents(docs)

        # ============================
        # 7. Add Chunks to the Vector Store
        # ============================

        # Store all the chunks as embeddings in the vector DB
        vector_store.add_documents(all_splits)

        # ============================
        # 8. Done!
        # ============================

        print(f"✅ Índice {collection_name} creado y guardado en: {persist_directory}")
