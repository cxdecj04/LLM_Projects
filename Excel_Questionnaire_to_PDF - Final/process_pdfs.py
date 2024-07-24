import os
import numpy as np
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import faiss
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pandas as pd
import docx
import pandas as pd

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as pdf:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_docx_text(docx_paths):
    text = ""
    for docx_path in docx_paths:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def get_excel_text(excel_paths):
    text = ""
    for excel_path in excel_paths:
        df = pd.read_excel(excel_path)
        for index, row in df.iterrows():
            text += " ".join([str(cell) for cell in row]) + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, index_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embedded_texts = [embeddings.embed_documents([chunk])[0] for chunk in text_chunks]
    embedded_texts_np = np.array(embedded_texts).astype('float32')
    d = embedded_texts_np.shape[1]  
    nlist = min(100, len(embedded_texts_np))  
    quantizer = faiss.IndexFlatL2(d) 
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.train(embedded_texts_np)
    index.add(embedded_texts_np)
    faiss.write_index(index, f"{index_name}_faiss_index_ivf")

def process_files(file_paths):
    all_text = ""
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            all_text += get_pdf_text([file_path])
        elif file_path.endswith(".docx"):
            all_text += get_docx_text([file_path])
        elif file_path.endswith(".xlsx"):
            all_text += get_excel_text([file_path])
    return all_text

def main():
    org_file_map = {
        "Org1": ["rfp1.pdf", ],
        "Org2": ["rfp2.pdf", ]
    }

    for org, file_list in org_file_map.items():
        all_text_chunks = []
        all_text = process_files(file_list)
        text_chunks = get_text_chunks(all_text)
        all_text_chunks.extend(text_chunks)
        
        with open(f"{org}_text_chunks.pkl", "wb") as f:
            pickle.dump(all_text_chunks, f)
        
        get_vector_store(all_text_chunks, org)
        print(f"Organization {org} files processed successfully.")

if __name__ == "__main__":
    main()
