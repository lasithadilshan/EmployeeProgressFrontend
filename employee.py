import streamlit as st
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


llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
embeddings = OpenAIEmbeddings()

file = "/Employee.pdf"
persist_directory = "app_db"



def process_file():
    loader = PyPDFLoader('Employee.pdf')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectordb = Chroma.from_documents(splits, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()

def read_doc_and_generate_response(prompt):
    prompt = f"""
Please generate a {prompt} Employee Evolution Report based on his evaluations contained in the PDF. The report should include the following sections:

1. Introduction: Brief overview of the purpose of this report and the evaluation period it covers, specifically for {prompt}.
2. Employee Summary: Detail {prompt}’s employee ID, positions held, and department assignments.
3. Performance Overview:
    - Summarize {prompt}'s performance ratings and supervisor comments.
    - Highlight his top performances and areas needing improvement.
4. Detailed Analysis:
    - Provide a detailed evaluation of {prompt} including strengths, areas for improvement, and recommended development actions.
    - Discuss any trends or patterns observed in his performance.
5. Conclusion: Summarize key findings and propose recommendations for {prompt}’s professional development.
6. Appendix: Include any additional notes or raw data excerpts that are relevant to {prompt}’s evaluations.

The report should be formal, well-organized, and insightful, leveraging {prompt}'s evaluation data to provide actionable insights to management.
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

def main():
    st.title("Employee Evaluation Report")

    # List of employee IDs for the dropdown menu
    employee_ids = ['E123', 'E124', 'E125']

    # Creating form for user input
    with st.form("emp_info_form"):
        emp_id = st.selectbox("Select Employee ID", employee_ids)

        # Form submission button
        submit_button = st.form_submit_button("Generate")

    if submit_button:
        process_file()
        response = read_doc_and_generate_response(emp_id)
        # report = generate_evaluation_report(emp_id)
        st.write("### Employee Evaluation Report")
        st.markdown(response)

if __name__ == "__main__":
    main()