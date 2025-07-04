import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

os.environ["CHROMA_TELEMETRY"] = "FALSE"

# Load from disk
laboral_vector_store = Chroma(
    collection_name="laboral",
    embedding_function=embeddings,
    persist_directory="/home/janf/Projects/legal_assistant_ai_agent/index/laboral/",  # Where to save data locally, remove if not necessary
)


laboral_retriever = laboral_vector_store.as_retriever()

civil_vector_store = Chroma(
    collection_name="civil",
    embedding_function=embeddings,
    persist_directory="/home/janf/Projects/legal_assistant_ai_agent/index/civil/",  # Where to save data locally, remove if not necessary
)

civil_retriever = civil_vector_store.as_retriever()

penal_vector_store = Chroma(
    collection_name="penal",
    embedding_function=embeddings,
    persist_directory="/home/janf/Projects/legal_assistant_ai_agent/index/penal/",  # Where to save data locally, remove if not necessary
)

penal_retriever = penal_vector_store.as_retriever()

general_vector_store = Chroma(
    collection_name="general",
    embedding_function=embeddings,
    persist_directory="/home/janf/Projects/legal_assistant_ai_agent/index/general/",  # Where to save data locally, remove if not necessary
)

general_retriever = general_vector_store.as_retriever()
