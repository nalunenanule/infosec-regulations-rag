from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class PDFUtilities:
    @staticmethod
    def load_and_split_pdfs(file_urls: List[str]) -> tuple[List[Document], int]:
        """
        Загружает PDF-документы из списка BytesIO и сразу разделяет их на чанки.

        :param files: список PDF-файлов в памяти
        :return: кортеж (список Document после сплита, количество файлов)
        """
        all_docs = []

        # Текстовый сплиттер
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=300,
            separators=["\n", "Статья ", "Раздел ", "Глава "]
        )

        for url in file_urls:
            loader = PyPDFLoader(url)
            docs = loader.load()
            split_docs = splitter.split_documents(docs)
            all_docs.extend(split_docs)

        return all_docs, len(file_urls)