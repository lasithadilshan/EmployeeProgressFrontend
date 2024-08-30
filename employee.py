import os
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from google.colab import userdata
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from tempfile import NamedTemporaryFile
from app_secrets import OPENAI_API_KEY

from typing import Union
from fastapi import FastAPI

app = FastAPI()

os.environ["OPENAI_API_KEY"]= "sk-proj-w57OAely6LYXd0x6tYu8T3BlbkFJISQrSx4nShTsEDjp190Z"
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
embeddings = OpenAIEmbeddings()

file = "/J01B5400028Attach5-BRD.pdf"
persist_directory = "app_db"



def process_file():
    loader = PyPDFLoader('J01B5400028Attach5-BRD.pdf')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectordb = Chroma.from_documents(splits, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()

def read_doc_and_generate_response(emp_id):
    prompt = f"""
     Think you as senior buissness analysis. Your responsibility is read the Buissness Requiremnt Document and Write the User Stories according to that BRD.
 Think step by step and write the all possible user stories to the Buissness Requiremnt Document.
"""
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k":4})
    chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever)
    request = {"question": prompt, "chat_history": []}
    result = chain(request)
    return result['answer']

def generate_evaluation_report(emp_id):
    # This function simulates generating a detailed report for the employee.
    # Real data processing and report generation would be implemented here.
    report = f"""
    **Employee Evaluation Report for ID {emp_id}**

    **Introduction:**
    - Overview of the evaluation period and objectives.

    **Employee Summary:**
    - Employee ID: {emp_id}
    - Position: Data Analyst
    - Department: Analytics

    **Performance Overview:**
    - Performance Rating: Excellent
    - Key Achievements: Successfully led the project on market analysis.
    - Areas for Improvement: Time management in multi-tasking scenarios.

    **Detailed Analysis:**
    - Strengths: Analytical skills, attention to detail, and leadership.
    - Areas for Improvement: Enhancing soft skills for better team collaboration.

    **Conclusion:**
    - The employee shows significant potential for leadership roles.
    - Recommended Actions: Enroll in advanced data analytics training.

    **Appendix:**
    - Additional notes or data excerpts can be added here.
    """
    return report


@app.get("/get_report/{emp_id}")
def read_root(emp_id: str):
    process_file()
    report = read_doc_and_generate_response(emp_id)
    return report