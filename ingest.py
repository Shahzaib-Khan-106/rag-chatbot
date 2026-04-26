import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def ingest_documents():
    print("Loading documents...")
    loader = DirectoryLoader(
        "./docs/",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    if not documents:
        print("No PDFs found in /docs folder!")
        return

    print(f"Loaded {len(documents)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding and storing in Chroma...")
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Done! Vector store saved to ./chroma_db")

if __name__ == "__main__":
    ingest_documents()