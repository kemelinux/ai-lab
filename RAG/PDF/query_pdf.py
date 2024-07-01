# deprecated
#from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma

# you can find other documents type loaders via the langchain website
# search for ¨Document loaders"

DATA_PATH = '/home/arjan/AI/RAG/PDF/Archibus'

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def get_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region="us-east-1"    
    )
    return embeddings

def get_ollama_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")   
    return embeddings

def add_to_chroma(chunks: list[Document]):
    db = Chrome(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    db.add_documents(new_chunks, ids=new_chunk_ids)
    db.persist()
    

documents = load_documents()
chunks = split_documents(documents)
print(chunks[0])
