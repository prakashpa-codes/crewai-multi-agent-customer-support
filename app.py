import os
from datetime import datetime
from pathlib import Path
from typing import Type

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import SerperDevTool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "shop_info.txt"
OUTPUT_DIR = BASE_DIR / "output"
ANSWERS_FILE = OUTPUT_DIR / "answers.txt"

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Whitecaps Cricket Gear & Equipment | Support",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
:root { color-scheme: light; }
.block-container { max-width: 920px; padding-top: 1rem; padding-bottom: 2rem; }
header[data-testid="stHeader"] { background: transparent; }
.brand { display:flex; align-items:center; gap:14px; padding:16px 18px; border:1px solid #e7e9ed;
         border-radius:18px; background:#ffffff; box-shadow:0 4px 18px rgba(20,35,55,.04); }
.brand-mark { width:48px; height:48px; border-radius:14px; display:flex; align-items:center; justify-content:center;
              background:#102a43; color:#fff; font-size:25px; flex-shrink:0; }
.brand-title { font-size:20px; font-weight:750; color:#102a43; line-height:1.2; }
.brand-subtitle { font-size:13px; color:#64748b; margin-top:5px; }
.status { margin-left:auto; white-space:nowrap; font-size:12px; color:#18794e;
          background:#eaf7ef; border-radius:999px; padding:6px 10px; }
.welcome { margin:20px 0 14px; }
.welcome h1 { font-size:26px; color:#102a43; letter-spacing:-.5px; margin-bottom:6px; }
.welcome p { color:#64748b; font-size:14px; }
.answer-label { font-size:12px; font-weight:750; text-transform:uppercase; letter-spacing:.06em; color:#64748b; margin:4px 0 8px; }
.footer-note { color:#64748b; font-size:12px; text-align:center; margin-top:24px; }
div[data-testid="stChatInput"] { border-radius:16px; }
[data-testid="stChatMessage"] { border-radius:16px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="brand">
      <div class="brand-mark">🏏</div>
      <div>
        <div class="brand-title">Whitecaps Cricket Gear &amp; Equipment</div>
        <div class="brand-subtitle">Cricket equipment advice and product support</div>
      </div>
      <div class="status">● Support assistant</div>
    </div>
    <div class="welcome">
      <h1>How can we help you today?</h1>
      <p>Ask about cricket bats, equipment, combo kits, prices, or choosing the right gear.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the product knowledge base…")
def build_vector_store(data_path: str, modified_time: float, api_key: str):
    # modified_time is part of the cache key so edits to shop_info.txt trigger re-indexing.
    loader = TextLoader(data_path, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=140)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=api_key,
    )
    return FAISS.from_documents(chunks, embeddings)


class SupportLogInput(BaseModel):
    customer_question: str = Field(..., description="The customer's original question.")
    source: str = Field(..., description="Whether the response used the shop knowledge base or web research.")
    answer: str = Field(..., description="The customer-facing answer to record.")


class AppendSupportLogTool(BaseTool):
    name: str = "append_support_interaction_to_text_file"
    description: str = (
        "Append the original customer question, response source, and customer-facing answer to "
        "output/answers.txt. Use this tool after preparing the response."
    )
    args_schema: Type[BaseModel] = SupportLogInput

    def _run(self, customer_question: str, source: str, answer: str) -> str:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        entry = (
            f"\n{'=' * 78}\n"
            f"Timestamp: {timestamp}\n"
            f"Customer question:\n{customer_question.strip()}\n\n"
            f"Response source: {source.strip()}\n\n"
            f"Customer-facing answer:\n{answer.strip()}\n"
        )
        with ANSWERS_FILE.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        return f"Successfully appended this interaction to {ANSWERS_FILE}."


def make_llm():
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return LLM(model=f"openai/{model}", api_key=os.environ["OPENAI_API_KEY"])


def needs_web_search(question: str) -> bool:
    """Route shop-specific questions to RAG; use web search for explicitly external/current questions."""
    q = question.lower()
    external_signals = (
        "latest", "breaking news", "current news", "recent news", "today's news", "this week",
        "in 2026", "released in", "newly released", "official rules", "international rules",
        "world cup", "ipl 2026", "live score", "live scores", "weather", "outside the shop",
        "compare with other brands", "compare other brands", "reviews online", "online reviews",
        "what is the latest", "latest update", "recent update", "who won", "upcoming matches",
        "upcoming tournament", "current rankings", "latest rankings", "news about",
    )
    general_research_signals = (
        "explain the difference", "what is the difference", "how does it work", "history of",
        "rules of cricket", "how to improve", "training tips", "technique", "recommend a brand",
    )
    if any(term in q for term in external_signals + general_research_signals):
        return True
    # If the question is about the shop's catalogue, prices, stock, policies, or hours, keep it local.
    local_signals = (
        "₹", "rs.", "rupees", "price", "cost", "budget", "under ", "between ", "less than ",
        "more than ", "available", "availability", "in stock", "stock", "catalogue", "catalog",
        "do you sell", "do you have", "your shop", "your store", "opening hours", "working hours",
        "business hours", "what time", "open on", "closed on", "shipping", "delivery", "return policy",
        "payment", "contact", "address", "opening", "closing", "sunday", "monday", "tuesday",
        "wednesday", "thursday", "friday", "saturday", "combo kit", "combo kits", "our products",
    )
    if any(term in q for term in local_signals):
        return False
    # Simple product enquiries should use the shop catalogue rather than unrelated public search results.
    product_terms = (
        "cricket bat", "bats", "bat", "helmet", "helmets", "pads", "pad", "gloves", "glove",
        "cricket ball", "balls", "kit bag", "kit bags", "shoes", "stumps", "thigh guard",
        "wicket keeping", "wicket-keeping", "equipment", "protective gear", "willow",
    )
    if any(term in q for term in product_terms):
        return False
    # Default to the shop knowledge base for ordinary customer support questions.
    return False


def run_support_crew(question: str, vector_store, use_web: bool):
    llm = make_llm()
    log_tool = AppendSupportLogTool()

    assistant = Agent(
        role="Shop Support Assistant",
        goal="Give a concise, accurate answer using the Whitecaps shop knowledge base.",
        backstory=(
            "You are the customer support assistant for Whitecaps Cricket Gear & Equipment. "
            "The supplied knowledge base is the source of truth for shop-specific demo catalogue facts. "
            "Never invent prices, products, stock, or policies. The catalogue is fictional demo data. "
            "Write in a natural shop-chat style: usually 2–5 sentences, or up to 5 bullets for product lists. "
            "Avoid long introductions and do not repeat the question."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    entry_agent = Agent(
        role="Entry Agent",
        goal="Record the customer's question and final answer in the local support log.",
        backstory=(
            "You accurately record the customer question, response source, and complete customer-facing answer. "
            "Do not rewrite the answer or claim that logging succeeded unless the tool succeeds."
        ),
        llm=llm,
        tools=[log_tool],
        verbose=False,
        allow_delegation=False,
    )

    matches = vector_store.similarity_search(question, k=4)
    retrieved_context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'shop knowledge base')}]\n{doc.page_content}"
        for doc in matches
    )

    task1 = Task(
        description=(
            "Customer question:\n{customer_question}\n\n"
            "Retrieved shop knowledge (RAG context):\n{retrieved_context}\n\n"
            "Answer using this context as the source of truth for shop-specific information. For product-filtering "
            "questions, compare the listed prices and show only matching products with their names and prices. "
            "Mention that catalogue prices and stock labels are fictional demo data when relevant. If the context "
            "does not contain the answer, say that you cannot confirm it from the shop information. Do not use web search. "
            "Keep the response concise: 2–5 sentences or a short list of up to 5 items."
        ),
        expected_output="A concise customer-facing answer grounded in the shop knowledge base.",
        agent=assistant,
    )

    agents = [assistant, entry_agent]
    tasks = [task1]
    response_source = "Shop knowledge base (RAG)"
    final_answer_task_context = [task1]

    if use_web:
        search_tool = SerperDevTool()
        web_assistant = Agent(
            role="Web Research Assistant",
            goal="Answer questions that require current or external public information.",
            backstory=(
                "You search the public web for current or general information when it is needed. "
                "Distinguish public information from Whitecaps' fictional shop catalogue; web results do not verify "
                "the shop's inventory or policies. Keep the answer concise, usually 2–5 sentences or up to 5 bullets."
            ),
            llm=llm,
            tools=[search_tool],
            verbose=False,
            allow_delegation=False,
        )
        task2 = Task(
            description=(
                "Customer question:\n{customer_question}\n\n"
                "Use public web search to answer the question requiring external/current information. "
                "Use reliable sources where possible. Do not claim that search results verify Whitecaps' stock, prices, "
                "or shop policies. Respond directly and concisely, with no long introduction."
            ),
            expected_output="A concise answer based on relevant public web information.",
            agent=web_assistant,
        )
        # For explicitly external questions, show the web-researched answer as the main response.
        log_task = Task(
            description=(
                "Use the append_support_interaction_to_text_file tool exactly once. Pass the original customer "
                "question as customer_question, source='Web research', and the complete Web Research Assistant "
                "answer as answer. Do not invent a successful log if the tool fails. After the tool call, briefly "
                "confirm the answer was prepared; do not repeat the full answer."
            ),
            expected_output="A brief log confirmation.",
            agent=entry_agent,
            context=[task2],
        )
        agents = [assistant, web_assistant, entry_agent]
        tasks = [task1, task2, log_task]
        response_source = "Web research"
        final_answer_task_context = [task2]
    else:
        log_task = Task(
            description=(
                "Use the append_support_interaction_to_text_file tool exactly once. Pass the original customer "
                "question as customer_question, source='Shop knowledge base (RAG)', and the complete output of the "
                "Shop Support Assistant task as answer. Do not rewrite the answer or claim logging succeeded unless "
                "the tool succeeds. After the tool call, briefly confirm the log was updated; do not repeat the answer."
            ),
            expected_output="A brief log confirmation.",
            agent=entry_agent,
            context=[task1],
        )
        tasks.append(log_task)

    crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False)
    result = crew.kickoff(inputs={"customer_question": question, "retrieved_context": retrieved_context})
    outputs = getattr(result, "tasks_output", []) or []
    # Task output 0 is RAG; task output 1 is web when web routing is active.
    rag_answer = str(outputs[0].raw).strip() if outputs else ""
    web_answer = str(outputs[1].raw).strip() if use_web and len(outputs) > 1 else ""
    answer = web_answer if use_web and web_answer else rag_answer
    return answer, response_source, rag_answer, web_answer


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🏏" if message["role"] == "assistant" else "🙂"):
        st.markdown(message["content"])
        if message.get("source"):
            st.caption(f"Source: {message['source']}")
        if message.get("details"):
            with st.expander("Additional context"):
                st.markdown(message["details"])

question = st.chat_input("Ask about cricket bats, kits, prices, or equipment…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(question)

    use_web = needs_web_search(question)
    required_keys = ["OPENAI_API_KEY"] + (["SERPER_API_KEY"] if use_web else [])
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        st.error(
            "Configuration is incomplete. Set the following environment variable(s) in your .env file and restart the app: "
            + ", ".join(missing)
        )
        st.stop()

    if not DATA_FILE.exists():
        st.error(f"Knowledge base not found: {DATA_FILE}")
        st.stop()

    try:
        spinner_text = "Checking the shop knowledge base and public web…" if use_web else "Checking the shop catalogue…"
        with st.spinner(spinner_text):
            store = build_vector_store(
                str(DATA_FILE),
                DATA_FILE.stat().st_mtime,
                os.environ["OPENAI_API_KEY"],
            )
            answer, source, rag_answer, web_answer = run_support_crew(question, store, use_web)

        if not answer:
            answer = "I couldn't prepare a response for that question. Please try rephrasing it."

        with st.chat_message("assistant", avatar="🏏"):
            st.markdown(answer)
            st.caption(f"Source: {source}. Support interaction is recorded in the local log when the Entry Agent succeeds.")
            # Keep RAG context available without making it compete with the main response.
            if use_web and rag_answer:
                with st.expander("Shop catalogue context"):
                    st.markdown(rag_answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "source": source,
            "details": rag_answer if use_web and rag_answer else "",
        })
    except Exception as exc:
        st.error(
            "The support assistant could not complete this request. Check your API keys, network connection, "
            "and terminal output, then try again."
        )
        with st.expander("Technical error details"):
            st.code(f"{type(exc).__name__}: {exc}")

st.markdown(
    '<div class="footer-note">Whitecaps Cricket Gear &amp; Equipment · Demo catalogue and sample support knowledge</div>',
    unsafe_allow_html=True,
)
