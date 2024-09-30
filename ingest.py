import os
import warnings
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Set up logging
logging.basicConfig(level=logging.INFO)

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")

# Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.
    This function loads data from PDF files, splits the documents into chunks,
    transforms them into embeddings using OllamaEmbeddings, and persists the embeddings 
    into a Chroma vector database.
    """
    logging.info("Starting the process to create vector database.")
    
    # Initialize PDF Loader
    pdf_loader = DirectoryLoader("data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    logging.info("PDF loader initialized.")
    
    loaded_documents = pdf_loader.load()
    logging.info(f"Loaded {len(loaded_documents)} documents from PDF files.")
    
    # Split loaded documents into chunks (optimized for mixed language)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    chunked_documents = text_splitter.split_documents(loaded_documents)
    
    logging.info(f"Split documents into {len(chunked_documents)} chunks.")
    
    # Add metadata to chunks for better query handling
    for doc in chunked_documents:
        doc.metadata["language"] = "mixed"  # Example: metadata can store "mixed", "Hindi", or "English"
    
    logging.info("Metadata added to document chunks.")
    
    # Initialize Ollama Embeddings (optimized for mixed language support)
    ollama_embeddings = OllamaEmbeddings(model="mistral")
    logging.info("Ollama embeddings initialized using 'mistral' model.")
    
    # Create and persist a Chroma vector database
    logging.info("Starting to embed documents and create Chroma vector database...")
    
    # Use from_texts method if from_documents causes issues
    vector_database = Chroma.from_texts(
        texts=[doc.page_content for doc in chunked_documents],  # Extract text from each document
        embedding_function=ollama_embeddings,
        persist_directory=DB_DIR,
    )
    
    vector_database.persist()
    logging.info("Vector database created and persisted.")

if __name__ == "__main__":
    create_vector_database()
