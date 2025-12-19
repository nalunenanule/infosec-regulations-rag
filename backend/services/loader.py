import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdfs() -> List[Document]:
    docs = []
    pdf_dir = "./data/pdf"
    files = os.listdir(pdf_dir)
    for file in files:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
            docs.extend(loader.load())
    return docs, len(files)

def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=300,
        separators=["\n", "Статья ", "Раздел ", "Глава "]
    )
    return splitter.split_documents(documents)