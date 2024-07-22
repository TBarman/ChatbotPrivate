import os
import streamlit as st
import requests
from dotenv import find_dotenv, load_dotenv

from ImageRead import GetTextRead
from Translation import Translate
from Translation import GetLanguage

import tempfile


from dotenv import load_dotenv
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








response = requests.get('https://httpbin.org/ip')
ip_address = response.json()['origin']

st.write(f'Public IP Address: {ip_address}')




















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


    # initializes chat log
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"speaker": "assistant", "text": "What would you like to ask?"}]

    # displays all messages in the chat log
    for message in st.session_state.messages:
        with st.chat_message(message["speaker"]):
            st.write(message["text"])

    # displays user input and records it in a variable
    if input := st.chat_input("Enter text here"): # if the user has entered something, assign it to the variable input and...
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

            db_user = os.getenv('DB_USER')
            db_password = os.getenv('DB_PASSWORD')
            db_name = os.getenv('DB_NAME')

            db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@localhost:3306/{db_name}")
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
                response = generate_response(input)

        st.session_state.messages.append({"speaker": "assistant", "text": response})
        # Say response
        with st.chat_message("assistant"):
            st.write(response)


if __name__ == '__main__':
    main()
