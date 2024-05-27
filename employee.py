import streamlit as st
import os
# import pdfkit
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
# from tempfile import NamedTemporaryFile
from fpdf import FPDF
from io import BytesIO

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

def read_doc_and_generate_response(emp_id):
    prompt = f"""
    Please generate an Employee Evolution Report for employee ID {emp_id}. This report should provide a comprehensive overview of the employee's performance and contributions over the evaluation period from January to December 2023.

    1. **Introduction**:
        - State the purpose of this report.
        - Discuss the evaluation period and its relevance.

    2. **Employee Summary**:
        - Employee ID: {emp_id}
        - Name: John Doe
        - Position: Software Engineer
        - Department: IT

    3. **Performance Overview**:
        - Performance Summary: Demonstrates excellent problem-solving abilities and punctuality in meeting deadlines.
        - Top Performances and Achievements: Highlighting the development and launch of X software.

    4. **Detailed Analysis**:
        - Strengths: Technical expertise and problem-solving.
        - Areas for Improvement: Communication and team collaboration.
        - Future Goals: Leading major projects and enhancing team engagement.
        - Professional Development: Impact of courses in machine learning and agile methodologies on his performance.

    5. **Conclusion**:
        - Summarize key findings and achievements.
        - Recommendations for further professional development.

    6. **Appendix**:
        - Include any additional relevant data or feedback excerpts.

    The report should be structured to provide actionable insights and detailed analysis of the employee's contributions and potential for future growth.

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

        # Create HTML content from the response
    html_content = f/"""
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
    </style>
    </head>
    <body>
        {report.replace('\n', '<br>')}
    </body>
    </html>
    """
    return html_content

st.title("Employee Evaluation Report")

# List of employee IDs for the dropdown menu
employee_ids = ['E123', 'E124', 'E125']

emp_id = st.selectbox("Select Employee ID", employee_ids)
submit_button = st.button("Generate Report")
download_btn = st.button("Download PDF")

if submit_button:
    # Generate the HTML content for the report
    if 'html_content' not in st.session_state:
        st.session_state.html_content = generate_evaluation_report(emp_id)

    st.write("### Employee Evaluation Report")
    st.markdown(st.session_state.html_content, unsafe_allow_html=True)
    

if download_btn:
    # Create instance of FPDF class
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    response_variable = "This is the content that will be written to the PDF."
    pdf.cell(200, 10, txt=response_variable, ln=True, align='C')

    # Save to a BytesIO object
    pdf_output = BytesIO()
    pdf.output(pdf_output, 'F')

    # Reset the buffer's cursor position
    pdf_output.seek(0)

    # Provide download button
    st.download_button(label="Download your PDF",
                       data=pdf_output,
                       file_name="response_variable.pdf",
                       mime='application/pdf')

