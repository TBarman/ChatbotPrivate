import os
import requests
from dotenv import find_dotenv, load_dotenv

from ImageRead import GetTextRead
from Translation import Translate
from Translation import GetLanguage

import tempfile

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

import pandas as pd
import pymysql



st.set_page_config(layout="wide")
load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

def generate_response(prompt):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": f"{prompt}",
    })

    try:
        generated_text = output[0]['generated_text']
    except KeyError:
        print('KeyError encountered when accessing "generated_text"')
        generated_text = 'KeyError encountered when accessing "generated_text"'

    return generated_text.replace(prompt, "").strip()



def main():
    #Streamlit
    st.title("Huggingface Chatbot")


    # sidebar for inputting an image

    with st.sidebar:
        temp_path = None
        if file := st.file_uploader("Please upload a file", type=["png", "pdf"]):
            if file.name.endswith(".png"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file.write(file.getvalue())
                    temp_path = temp_file.name
                st.image(file)
                st.success("Image successfully downloaded!")

            elif file.name.endswith(".pdf"):
                # copied from https://youtu.be/Dh0sWMQzNH4
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # split into chunks
                char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000,
                                                           chunk_overlap=200, length_function=len)
                text_chunks = char_text_splitter.split_text(text)

                # create embeddings
                embeddings = OpenAIEmbeddings()
                docsearch = FAISS.from_texts(text_chunks, embeddings)
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                st.success("PDF successfully downloaded!")

    col1, col2 = st.columns(2, gap='medium')
    # initializes chat log
    input = st.chat_input("Enter text here")
    with col1:
        with st.container():
            if "messages" not in st.session_state.keys():
                st.session_state.messages = [{"speaker": "assistant", "text": "What would you like to ask?"}]

            # displays all messages in the chat log
            for message in st.session_state.messages:
                with st.chat_message(message["speaker"]):
                    st.write(message["text"])



            # displays user input and records it in a variable
            if input != None: # if the user has entered something, assign it to the variable input and...
                st.session_state.messages.append({"speaker": "user", "text": input})
                with st.chat_message("user"):
                    st.write(input)

            # generates and displays chatbot response
            if st.session_state.messages[-1]["speaker"] != "assistant": # if the speaker of the last message was not the assistant

                # If the user asked for the bot to read a picture
                if st.session_state.messages[-1]["text"] == "Can you read the image for me?":
                    if file != None:
                        if file.name.endswith(".png"):
                            with st.spinner("Thinking..."):
                                text = GetTextRead(temp_path)
                                if (lang := GetLanguage(text)) != "en":
                                    text = Translate(text, lang)
                                response = text
                    else:
                        response = "there is no image"

                # If the user is trying to ask questions about a pdf
                elif st.session_state.messages[-1]["text"][:5] == "PDF: ":
                    if file.name.endswith(".pdf"):
                        with st.spinner("Thinking..."):
                            query = st.session_state.messages[-1]["text"][5:]
                            docs = docsearch.similarity_search(query)
                            response = chain.run(input_documents=docs, question=query)
                    else:
                        response = "there is no pdf"

                elif st.session_state.messages[-1]["text"][:7] == "QUERY: ":
                    # from https://medium.com/@koratarpans99/natural-language-to-sql-with-langchain-nl2sql-f4adc84b81da

                    with st.spinner("Thinking..."):
                        # local
                        db_user = os.getenv('DB_USER')
                        db_password = os.getenv('DB_PASSWORD')
                        db_name = os.getenv('DB_NAME')
                        # db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@localhost:3306/{db_name}")

                        # Azure
                        db_user_azure = os.getenv('DB_USER_AZURE')
                        db_password_azure = os.getenv('DB_PASSWORD_AZURE')
                        db_server_name = os.getenv('DB_SERVER_NAME')
                        db_host = os.getenv('DB_HOST')
                        db_port = os.getenv("DB_PORT")
                        db_name_azure = os.getenv("DB_NAME_AZURE")
                        db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user_azure}:{db_password_azure}@{db_host}:{db_port}/{db_name_azure}")
                        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                        generate_query = create_sql_query_chain(llm, db)

                        execute_query = QuerySQLDataBaseTool(db=db)
                        answer_prompt = PromptTemplate.from_template(
                            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
                            Question: {question}
                            SQL Query: {query}
                            SQL Result: {result}
                            Answer: """
                        )
                        rephrase_answer = answer_prompt | llm | StrOutputParser()
                        chain = (
                                RunnablePassthrough.assign(query=generate_query).assign(
                                    result=itemgetter("query") | execute_query
                                )
                                | rephrase_answer
                        )
                        response = chain.invoke({"question": st.session_state.messages[-1]["text"][7:]})
                # Otherwise
                else:
                    with st.spinner("Thinking..."):
                        response = generate_response(st.session_state.messages[-1]["text"])

                st.session_state.messages.append({"speaker": "assistant", "text": response})
                # Say response

                for message in st.session_state.messages:
                    print(message["text"])
                with st.chat_message("assistant"):
                    st.write(response)


    with col2:
        hello = st.text_input("hello")
        table_name = st.text_input("What is the table name?", value="data")
        st.session_state.rows = st.number_input("How many rows?", value=1)
        st.session_state.columns = st.columns(st.number_input("How many columns?", value=2))

        with st.form(key='my_form'):
            row_names = [f"Row {i + 1}" for i in range(st.session_state.rows)]
            column_names = [""] * len(st.session_state.columns)

            data_list = [[None for c in range(len(column_names))] for r in range(len(row_names))]

            for c in range(len(st.session_state.columns)):
                with st.session_state.columns[c]:
                    column_names[c] = st.text_input(f"Column {c + 1} Name")
                    for r in range(st.session_state.rows):
                        data_list[r][c] = st.text_input(f"({r + 1}, {c + 1})")

            if st.form_submit_button():
                data = pd.DataFrame(data_list, columns=column_names)
                st.table(data)
                db_user_azure = os.getenv('DB_USER_AZURE')
                db_password_azure = os.getenv('DB_PASSWORD_AZURE')
                db_host = os.getenv('DB_HOST')
                db_name_azure = os.getenv("DB_NAME_AZURE")
                connection = pymysql.connect(
                    host=db_host,
                    user=db_user_azure,
                    password=db_password_azure,
                    database=db_name_azure
                )
                if connection.is_connected():
                    cursor = connection.cursor()

                    # SQL command to create a table
                    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
                    for c in range(len(column_names)):
                        if c < len(column_names) - 1:
                            create_table_query += column_names[c] + " VARCHAR(50),"
                        else:
                            create_table_query += column_names[c] + " VARCHAR(50));"

                    # Execute the SQL command
                    cursor.execute(create_table_query)
                    connection.commit()

                    insert_table_query = f"INSERT INTO {table_name} ("
                    for c in range(len(column_names)):
                        if c < len(column_names) - 1:
                            insert_table_query += column_names[c] + ", "
                        else:
                            insert_table_query += column_names[c] + ")"
                    insert_table_query += "VALUES ("
                    for c in range(len(column_names)):
                        if c < len(column_names) - 1:
                            insert_table_query += "%s, "
                        else:
                            insert_table_query += "%s)"

                    data_tuples = [tuple(row) for row in data_list]

                    cursor.executemany(insert_table_query, data_tuples)
                    connection.commit()

                    cursor.close()
                    connection.close()

                    st.success("Table created successfully")


if __name__ == '__main__':
    main()
