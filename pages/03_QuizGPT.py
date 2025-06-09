# Implement QuizGPT but add the following features:

# Use function calling.
# Allow the user to customize the difficulty of the test and make the LLM generate hard or easy questions.
# Allow the user to retake the test if not all answers are correct.
# If all answers are correct use st.ballons.
# Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar
# Using st.sidebar put a link to the Github repo with the code of your Streamlit app.

# Answer: https://github.com/fullstack-gpt-python/assignment-16/blob/main/app.py

import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnablePassthrough
import streamlit as st
from langchain.retrievers import WikipediaRetriever

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

if "retry" not in st.session_state:
    st.session_state.retry = False

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


def format_docs(inputs):
    docs = inputs["input"]
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context, please make several quiz (at least 5 quiz) to test the user's knowledge about the text.
    
    Please adjust the quiz level according to the user's difficulty level(Easy, Hard) request.

    Each quiz should have 4 answers, three of them must be incorrect and one should be correct.

    Here are informations. Please make quiz.
         
    Difficulty level: {level}

    Context: {context}
""",
        )
    ]
)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, level):
    return chain.invoke({"level": level, "input": _docs})


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    api_key = st.text_input("Insert your OpenIA API key")
    st.markdown("---")
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
    st.markdown("---")
    level = st.selectbox(
        "Select the problem difficulty.",
        (
            "Easy",
            "Hard",
        ),
    )
    st.markdown(
        "[Github Repository](https://github.com/hi-jason-jung/gpt-project/blob/master/pages/03_QuizGPT.py)"
    )

if api_key:
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )

    chain = {"context": format_docs, "level": RunnablePassthrough()} | prompt | llm

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name, level)
    response = json.loads(response.additional_kwargs["function_call"]["arguments"])

    is_perfect = True
    with st.form("questions_form"):
        for idx, question in enumerate(response["questions"]):
            st.write(f"{idx+1}. {question['question']}")
            value = st.radio(
                "Choose the answer.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )

            # if not st.session_state.retry:
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
                is_perfect = False
        button = st.form_submit_button()

    if is_perfect and button:
        st.session_state.retry = False
        st.balloons()
