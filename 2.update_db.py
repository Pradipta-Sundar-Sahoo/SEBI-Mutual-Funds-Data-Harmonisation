import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from extract_amendments0 import main

OPENAI_API_KEY = "53ddfdcdc586463ba277952a1cf23fe2"
os.environ["OPENAI_API_VERSION"] = "2024-05-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://tecosys.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    openai_api_version=os.environ["OPENAI_API_VERSION"],
)

faiss_db = FAISS.load_local("main_db_open", embeddings, allow_dangerous_deserialization=True)

with open("SEBI Grievance Regulations 2023 2nd pdf.pdf", "rb") as f:
    texts = main(f)
    print("Relevant Text fetched:")
    print(texts)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
new_chunks = text_splitter.split_documents([Document(page_content=text) for text in texts])

chunk_texts = [chunk.page_content for chunk in new_chunks]

new_embeddings = embeddings.embed_documents(chunk_texts)

faiss_db.add_documents(new_chunks)

faiss_db.save_local("main_db_open")
print("fetched text is added to master db")