from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import SystemMessage
import streamlit as st

if "messages" not in st.session_state:
    st.session_state["messages"] = []


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        if self.message:
            save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        print("test - message", message)
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def load_memory(_):
    messages = st.session_state.get("messages", [])
    return "\n".join([f"{m['role']}: {m['message']}" for m in messages])


# --- Wikipedia Search Tool ---
class WikipediaSearchArgs(BaseModel):
    query: str = Field(description="Search query for Wikipedia.")


class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearch"
    description = "Use this tool to search and get a summary from Wikipedia."
    args_schema: Type[WikipediaSearchArgs] = WikipediaSearchArgs

    def _run(self, query: str):
        try:
            wiki = WikipediaAPIWrapper()
            return wiki.run(query)
        except Exception as e:
            return f"Wikipedia error: {e}"


# --- DuckDuckGo Search Tool ---
class DuckDuckGoSearchArgs(BaseModel):
    query: str = Field(description="Search query for DuckDuckGo.")


class DuckDuckGoSearchTool(BaseTool):
    name = "DuckDuckGoSearch"
    description = "Use this tool to search and get a summary from DuckDuckGo."
    args_schema: Type[DuckDuckGoSearchArgs] = DuckDuckGoSearchArgs

    def _run(self, query: str):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)


# --- Web Scraping Tool ---
class WebScrapingArgs(BaseModel):
    url: str = Field(description="The full URL of the web page to scrape.")


class WebScrapingTool(BaseTool):
    name = "WebScrapingTool"
    description = "Extracts visible text from a given web page"
    args_schema: Type[WebScrapingArgs] = WebScrapingArgs

    def _run(self, url: str):
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            combined_text = "\n".join([doc.page_content for doc in docs])
            return combined_text[:3000]  # LLM 입력 제한 고려
        except Exception as e:
            return f"Web scraping failed: {e}"


# --- Save to Text File Tool ---
class SaveToTextFileArgs(BaseModel):
    content: str = Field(description="The research content to save.")


class SaveToTextFileTool(BaseTool):
    name = "SaveToTextFile"
    description = "Save the research result to a .txt file"
    args_schema: Type[SaveToTextFileArgs] = SaveToTextFileArgs

    def _run(self, content: str):
        with open("research_output.txt", "w", encoding="utf-8") as f:
            f.write(content)
        return "Research saved to research_output.txt"


# 5. 에이전트 초기화
tools = [
    WikipediaSearchTool(),
    DuckDuckGoSearchTool(),
    WebScrapingTool(),
    SaveToTextFileTool(),
]


st.set_page_config(
    page_title="WebSearchGPT_Agent",
    page_icon="💼",
)

st.markdown(
    """
    # WebSearchGPT_Agent
            
    Welcome to WebSearchGPT_Agent.
            
    Ask me your questions. I will search your query on the website and organize it for you.
"""
)

with st.sidebar:
    api_key = st.text_input("Insert your OpenIA API key")
    st.markdown(
        "[Github Repository - WebSearchGPT_Agent](https://github.com/hi-jason-jung/gpt-project/blob/master/pages/07_WebSearchGPT_Agent.py)"
    )

if not api_key:
    st.error("Please input your OpenAI API Key on the sidebar")
else:
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.1,
        model_name="gpt-4o-mini",
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

    system_message = SystemMessage(
        content="""
        You are a research expert.

        Your task is to use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided. 

        When you find a relevant website through DuckDuckGo, you must scrape the content from that website. Use this scraped content to thoroughly research and formulate a detailed answer to the question. 

        Combine information from Wikipedia, DuckDuckGo searches, and any relevant websites you find. Ensure that the final answer is well-organized and detailed, and include citations with links (URLs) for all sources used.

        Your research should be saved to a .txt file, and the content should match the detailed findings provided. Make sure to include all sources and relevant information.

        The information from Wikipedia must be included.

        Ensure that the final .txt file contains detailed information, all relevant sources, and citations.
        """
    )

    agent = initialize_agent(
        llm=llm,
        tools=tools,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs={"system_message": system_message},
    )

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask me what you wnat to know about any stock.")

    if message:
        send_message(message, "human")
        with st.chat_message("ai"):
            agent.run(message)
