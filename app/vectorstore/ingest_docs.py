import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

os.environ["CHROMA_TELEMETRY"] = "FALSE"


    
for dirpath, dirnames, filenames in os.walk("/home/janf/Projects/legal_assistant_ai_agent/docs"):
    parent_dir = dirpath.split('/')[-1]
    for filename in filenames:
        pdf_path = os.path.join(dirpath, filename)

        # 2. Directorio donde se guardará el índice
        persist_directory = f"index/{parent_dir}/"

        # 3. Nombre de la colección (asociado a la categoría)
        collection_name = parent_dir

        # 4. Cargar y dividir el PDF en documentos
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # 5. Crear embeddings (usa tu clave OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings()

        # 6. Crear índice con Chroma
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,  # Where to save data locally, remove if not necessary
        )

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        print(f"✅ Índice {collection_name} creado y guardado en: {persist_directory}")
