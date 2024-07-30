import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
OPENAI_API_KEY = "your_api_key"
os.environ["OPENAI_API_VERSION"] = "OPENAI_API_VERSION"
os.environ["AZURE_OPENAI_ENDPOINT"] = "AZURE_OPENAI_ENDPOINT"
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
file_path = os.path.join(os.path.dirname(__file__), 'main.pdf')
loader = PyPDFLoader(file_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
embeddings=AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    openai_api_version=os.environ["OPENAI_API_VERSION"],
)
faiss_db = FAISS.from_documents(texts, embeddings)
faiss_db.save_local("main_db_open")
print("Master Vector db Created")
