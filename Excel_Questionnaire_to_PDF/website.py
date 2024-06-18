import streamlit as st
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

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
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
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.title("PDF and Excel Processor")

    st.subheader("Step 1: Upload PDF Files")
    pdf_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    st.subheader("Step 2: Upload Excel File")
    excel_file = st.file_uploader("Choose an Excel file", type="xlsx")
    
    if st.button("Process Files"):
        if pdf_files and excel_file:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed successfully.")
            
            with st.spinner("Processing Excel file..."):
                df = pd.read_excel(excel_file)
                if 'Question' in df.columns:
                    df['Answer'] = df['Question'].apply(user_input)
                    output_excel = "output_with_answers.xlsx"
                    df.to_excel(output_excel, index=False)
                    st.success("Excel file processed successfully.")
                    
                    with open(output_excel, "rb") as file:
                        st.download_button(
                            label="Download Processed Excel File",
                            data=file,
                            file_name=output_excel,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.error("The Excel file does not contain a 'Question' column.")
        else:
            st.error("Please upload both PDF and Excel files.")

if __name__ == "__main__":
    main()
