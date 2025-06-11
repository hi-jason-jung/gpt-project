# - Build a SiteGPT version for Cloudflare's documentation.
# - The chat bot should be able to answers questions about the documentation of each one of these products:
#   - AI Gateway
#   - Cloudflare Vectorize
#   - Workers AI
# - Use the sitemap(https://developers.cloudflare.com/sitemap-0.xml) to find all the documentation pages for each product.
# - Your submission will be tested with the following questions:
#   - "What is the price per 1M input tokens of the llama-2-7b-chat-fp16 model?"
#   - "What can I do with Cloudflare’s AI Gateway?"
#   - "How many indexes can a single account have in Vectorize?"
# - Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar
# - Using st.sidebar put a link to the Github repo with the code of your Streamlit app.

# Answer: https://github.com/fullstack-gpt-python/assignment-17/blob/main/app.py

from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import hashlib

if "messages" not in st.session_state:
    st.session_state["messages"] = []


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url, api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    safe_cache_filename = hashlib.md5(url.encode()).hexdigest()
    cache_dir = LocalFileStore(f"./.cache/{safe_cache_filename}/")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    vector_store = FAISS.from_documents(docs, cached_embeddings)

    return vector_store.as_retriever()


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    This GPT is specified for https://developers.cloudflare.com website.
            
    Please insert your openai api key on the sidebar and ask me your questions.
"""
)


with st.sidebar:
    api_key = st.text_input("Insert your OpenIA API key")
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
        value="https://developers.cloudflare.com/sitemap-0.xml",
        disabled=True,
    )
    st.markdown("---")
    st.markdown(
        "[Github Repository - SiteGPT](https://github.com/hi-jason-jung/gpt-project/blob/siteGPT/app.py)"
    )


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.1,
    )
    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    if not api_key:
        st.error("Please input your OpenAI API Key on the sidebar")
    else:
        retriever = load_website(url, api_key)
        send_message(
            "Hello! I can help you about 1) AI Gateway, 2) Cloudflare Vectorize and 3) Workers AI. Ask away!",
            "ai",
            save=False,
        )
        paint_history()
        message = st.chat_input("Ask a question to the website.")

        if message:
            send_message(message, "human")
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            with st.chat_message("ai"):
                chain.invoke(message)
