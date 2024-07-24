import os
import numpy as np
import pandas as pd
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import faiss
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
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
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context, give answer related to the information in the selected section in the question of the context\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, text_chunks, index_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index = faiss.read_index(f"{index_name}_faiss_index_ivf")
    query_embedding = embeddings.embed_query(user_question)
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=5)
    docs = [Document(page_content=text_chunks[i]) for i in I[0]]
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main(excel_path, output_excel):
    print("Processing Excel file...")
    df = pd.read_excel(excel_path)
    
    print("DataFrame loaded:")
    print(df.head())

    if 'Question' in df.columns and 'OrganizationID' in df.columns:
        text_chunks_dict = {}
        for org in df['OrganizationID'].unique():
            with open(f"{org}_text_chunks.pkl", "rb") as f:
                text_chunks_dict[org] = pickle.load(f)

        print("Applying user_input function...")
        df['Answer'] = df.apply(lambda row: user_input(row['Question'], text_chunks_dict[row['OrganizationID']], row['OrganizationID']), axis=1)
        print("Writing to Excel file...")
        df.to_excel(output_excel, index=False)
        print(f"Excel file processed successfully. Answers saved to {output_excel}")
    else:
        print("Required columns 'Question' and 'OrganizationID' not found in Excel file.")

if __name__ == "__main__":
    excel_path = "ques2copycopy.xlsx"
    output_excel = "output_with_answers.xlsx"
    main(excel_path, output_excel)
