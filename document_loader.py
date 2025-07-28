import os
import tempfile
import zipfile
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader, UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader, UnstructuredHTMLLoader,
    UnstructuredFileLoader, TextLoader, CSVLoader,
    UnstructuredImageLoader
)
# ✅ CHANGED: Switched to a more effective text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from pdf2image import convert_from_path
from PIL import Image
import pytesseract

load_dotenv()

def load_file_to_vectorstore(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"❌ File '{file_path}' is empty.")

    loader = None
    documents = [] # Initialize documents list

    try:
        if ext == ".pdf":
            try:
                # Try standard loader
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                if not documents or not any(doc.page_content.strip() for doc in documents):
                    raise Exception("Empty pages or scanned PDF")
            except:
                print("📸 Detected scanned PDF. Applying OCR...")
                documents = scanned_pdf_to_documents(file_path)

        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(file_path)
        elif ext in [".pptx", ".ppt"]:
            loader = UnstructuredPowerPointLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        elif ext in [".html", ".htm"]:
            loader = UnstructuredHTMLLoader(file_path)
        elif ext in [".jpg", ".jpeg", ".png"]:
            loader = UnstructuredImageLoader(file_path)
        elif ext == ".csv":
            return handle_csv(file_path)
        elif ext == ".zip":
            return handle_zip(file_path)
        else:
            raise ValueError(f"❌ Unsupported file type: {ext}")

        if loader:
            documents = loader.load()

        if not documents:
            raise ValueError(f"❌ No content extracted from '{file_path}'.")

    except Exception as e:
        raise RuntimeError(f"❌ Failed to load document '{file_path}': {e}")

    return convert_to_vectorstore(documents)

def convert_to_vectorstore(documents):
    # ✅ CHANGED: Using a more robust splitter with better overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documents)

    if not docs:
        raise ValueError("❌ No chunks created from the document.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def handle_zip(file_path):
    all_docs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        for root, _, files in os.walk(temp_dir):
            for name in files:
                full_path = os.path.join(root, name)
                try:
                    if os.path.getsize(full_path) == 0:
                        print(f"⚠️ Skipping empty file: {name}")
                        continue
                    # This now correctly calls the main loading function
                    vectorstore = load_file_to_vectorstore(full_path) 
                    # A more robust way to get all docs from a vectorstore
                    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'fetch_k': 50})
                    docs = retriever.get_relevant_documents(" ") # Get all documents
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"⚠️ Skipping {name}: {e}")

    if not all_docs:
        raise ValueError("❌ No valid documents found in ZIP file.")

    return convert_to_vectorstore(all_docs)


def handle_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("CSV is empty.")
        summary = df.describe(include="all").to_string()
        document = Document(page_content=summary)
        return convert_to_vectorstore([document])
    except Exception as e:
        raise RuntimeError(f"❌ Failed to process CSV: {e}")

def scanned_pdf_to_documents(file_path):
    """Convert scanned PDF pages into text using OCR"""
    images = convert_from_path(file_path)
    documents = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        if text.strip():
            documents.append(Document(page_content=text, metadata={"page": i + 1}))
    if not documents:
        raise ValueError("❌ OCR failed to extract any content from scanned PDF.")
    return documents

# ✅ Aliases for compatibility
load_to_vectorstore = load_file_to_vectorstore
load_pdf_to_vectorstore = load_file_to_vectorstore