import json
import openai
import streamlit as st
import time

from langchain.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain.document_loaders import WebBaseLoader

# Streamlit 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None

# Streamlit 페이지 설정
st.set_page_config(page_title="WebSearchGPT_Assistant", page_icon="🔍")

st.markdown(
    """
# WebSearchGPT_Assistant  
Welcome to the Assistant-powered research tool.  
Ask me a question, and I will research it using Wikipedia, DuckDuckGo, and Web Scraping.
"""
)

with st.sidebar:
    api_key = st.text_input("Insert your OpenAI API Key", type="password")
    st.markdown(
        "[GitHub Repository - WebSearchGPT_Assistant](https://github.com/hi-jason-jung/gpt-project/blob/master/pages/08_WebSearchGPT_Assistant.py)"
    )

if not api_key:
    st.error("Please input your OpenAI API Key on the sidebar")
    st.stop()

openai.api_key = api_key


# ---- Tool 실행 함수들 ----
def wikipedia_search(query):
    try:
        wiki = WikipediaAPIWrapper()
        return wiki.run(query)
    except Exception as e:
        return f"Wikipedia error: {e}"


def duckduckgo_search(query):
    try:
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)
    except Exception as e:
        return f"DuckDuckGo error: {e}"


def web_scraping(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs[0].page_content[:3000] if docs else "No content found"
    except Exception as e:
        return f"Web scraping failed: {e}"


def save_text_to_file(content):
    try:
        with open("research_output.txt", "w", encoding="utf-8") as f:
            f.write(content)
        return "Research saved to research_output.txt"
    except Exception as e:
        return f"File save error: {e}"


# ---- Assistant 생성 ----
if "assistant_id" not in st.session_state:
    assistant = openai.beta.assistants.create(
        name="WebSearch Assistant",
        instructions="""
        You are a research expert.

        Your task is to use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided. 

        When you find a relevant website through DuckDuckGo, you must scrape the content from that website. Use this scraped content to thoroughly research and formulate a detailed answer to the question. 

        Combine information from Wikipedia, DuckDuckGo searches, and any relevant websites you find. Ensure that the final answer is well-organized and detailed, and include citations with links (URLs) for all sources used.

        Your research should be saved to a .txt file, and the content should match the detailed findings provided. Make sure to include all sources and relevant information.

        The information from Wikipedia must be included.

        Ensure that the final .txt file contains detailed information, all relevant sources, and citations.
        """,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "wikipedia_search",
                    "description": "Search and get summary from Wikipedia.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "duckduckgo_search",
                    "description": "Search the web using DuckDuckGo.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_scraping",
                    "description": "Extracts visible text from a given web page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "save_text_to_file",
                    "description": "Save research results to a .txt file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"],
                    },
                },
            },
        ],
        model="gpt-4o-mini",
    )
    st.session_state["assistant_id"] = assistant.id


# ---- 대화 히스토리 렌더링 ----
def render_history():
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["message"])


def save_message(role, message):
    st.session_state["messages"].append({"role": role, "message": message})


def save_to_text_file(content):
    with open("research_output.txt", "w", encoding="utf-8") as f:
        f.write(content)
    return "✅ Research saved to research_output.txt"


render_history()
user_input = st.chat_input("What would you like to know?")

# ---- 사용자 입력 처리 ----
if user_input:
    save_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Thread 생성 또는 재사용
    if not st.session_state["thread_id"]:
        thread = openai.beta.threads.create()
        st.session_state["thread_id"] = thread.id
    else:
        thread = openai.beta.threads.retrieve(st.session_state["thread_id"])

    # 사용자 메시지 추가
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input,
    )

    # Assistant 실행
    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=st.session_state["assistant_id"],
    )

    with st.chat_message("ai"):
        message_box = st.empty()
        full_response = ""

        while True:
            run_status = openai.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            print(f"Run status: {run_status.status}")  # Debug log

            if run_status.status == "completed":
                break
            elif run_status.status == "requires_action":
                print("Tool call required. Preparing to run tools...")
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for call in tool_calls:
                    name = call.function.name
                    args = json.loads(call.function.arguments)
                    print(f"Calling tool: {name} with args: {args}")
                    if name == "duckduckgo_search":
                        result = duckduckgo_search(args["query"])
                    elif name == "wikipedia_search":
                        result = wikipedia_search(args["query"])
                    elif name == "web_scraping":
                        result = web_scraping(args["url"])
                    elif name == "save_text_to_file":
                        result = save_text_to_file(args["content"])
                    else:
                        result = "No such tool."

                    print(f"Tool result: {result[:4]}...")
                    tool_outputs.append({"tool_call_id": call.id, "output": result})

                run = openai.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )
            elif run_status.status in ["failed", "cancelled", "expired"]:
                message_box.markdown("⚠️ Assistant run failed.")
                st.stop()
            time.sleep(1)

        # 최종 메시지 출력
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        latest = messages.data[0].content[0].text.value
        full_response += latest
        message_box.markdown(full_response)
        save_message("ai", full_response)
