# pip install streamlit 
# pip install google-generativeai 
# pip install python-dotenv 
# pip install langchain 
# pip install PyPDF2 
# pip install chromadb 
# pip install faiss-cpu 
# pip install langchain_google_genai
# pip install -q -U google-generativeai
# pip install langchain_community
from langchain_community.vectorstores import FAISS
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd


# from google.colab import userdata

# GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

# genai.configure(api_key=GOOGLE_API_KEY)
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

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0000001)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main(pdf_paths, excel_path, output_excel):
    print("Processing PDFs...")
    raw_text = get_pdf_text(pdf_paths)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    print("PDFs processed successfully.")

    print("Processing Excel file...")
    df = pd.read_excel(excel_path)
    if 'Question' in df.columns:
        df['Answer'] = df['Question'].apply(user_input)
        df.to_excel(output_excel, index=False)
        print(f"Excel file processed successfully. Answers saved to {output_excel}")

if __name__ == "__main__":

    pdf_paths = ["yor_pdf_file.pdf"]
    excel_path = "your_excel_file.xlsx"
    output_excel = "output_with_answers.xlsx"

    main(pdf_paths, excel_path, output_excel)
