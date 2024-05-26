import streamlit as st
import os
import pdfkit
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

def main():
    st.title("Employee Evaluation Report")

    # List of employee IDs for the dropdown menu
    employee_ids = ['E123', 'E124', 'E125']

    # Creating form for user input
    with st.form("emp_info_form"):
        emp_id = st.selectbox("Select Employee ID", employee_ids)
        submit_button = st.form_submit_button("Generate Report")

    if submit_button:
        # Generate the report content
        report_content = generate_evaluation_report(emp_id)
        st.write("### Employee Evaluation Report")
        st.markdown(report_content)

        # Generate PDF from report content and create download button
        html_content = """
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
            {report_content.replace('\n', '<br>')}
        </body>
        </html>
        """
        pdf = pdfkit.from_string(html_content, False)
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf)
            tmpfile.flush()
            tmpfile.close()
            with open(tmpfile.name, "rb") as f:
                # Streamlit method to create a download button immediately
                st.download_button(
                    label="Download PDF",
                    data=f.read(),
                    file_name=f"Employee_{emp_id}_Evaluation_Report.pdf",
                    mime="application/pdf"
                )
            os.unlink(tmpfile.index)



if __name__ == "__main__":
    main()