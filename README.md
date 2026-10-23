# CrewAI Multi-Agent Customer Support

A sequential, three-agent customer-support application built with CrewAI and Streamlit. The first agent uses retrieval-augmented generation (RAG) over a local text knowledge base with FAISS; the second searches the public web; the third records the interaction.

## Demo

https://github.com/user-attachments/assets/6ea698e7-2b4a-42b8-87c0-76122925e595

## Architecture

```text
Customer query
     |
     v
Assistant (RAG)
  TextLoader -> text splitter -> OpenAI embeddings -> FAISS similarity search
     |
     v
Web Search Assistant
  Serper web search
     |
     v
Entry Agent
  Records the question and both answers in output/answers.txt
     |
     v
Streamlit chat interface
```

CrewAI runs the three agents using `Process.sequential`. The RAG index is built from `data/shop_info.txt` and cached for the Streamlit session; edits to the source file trigger a rebuild. The vector store is FAISS, not Chroma.

## Technology

- Python
- CrewAI sequential crews and tasks
- Streamlit chat interface
- LangChain `TextLoader` and `RecursiveCharacterTextSplitter`
- OpenAI embeddings (`text-embedding-3-small` by default)
- FAISS vector similarity search
- Serper web search through `crewai-tools`

## Demo case study: Whitecaps Cricket Gear & Equipment

The sample knowledge base models a cricket-only retail shop. It includes fictional bats, helmets, combo kits, protective equipment, cricket balls, bags, footwear, accessories, and demo prices. This makes it possible to test queries such as “Show English willow bats between ₹10,000 and ₹15,000.”

All product prices and stock labels are fictional demonstration data. They are not verified retail offers or live inventory. The sample knowledge base intentionally does not invent official delivery, return, refund, or warranty policies.

## Requirements

- Python 3.10 or newer (64-bit recommended)
- An OpenAI API key with access to the configured chat and embedding models
- A Serper API key

## Setup on Windows PowerShell

Run these commands from the project directory.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks virtual-environment activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Edit `.env` and replace the placeholders:

```dotenv
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Do not commit `.env`. API keys are read from environment configuration and must remain private.

## Run

```powershell
streamlit run app.py
```

Open the local address printed by Streamlit, normally `http://localhost:8501`. Stop the application with `Ctrl+C`.

## Configuration

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI chat and embedding access |
| `SERPER_API_KEY` | Serper web search |
| `OPENAI_MODEL` | CrewAI chat model; default `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model; default `text-embedding-3-small` |

## Project layout

```text
crewai-multi-agent-customer-support/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── shop_info.txt
├── screenshots/
└── output/                 # Created at runtime; excluded from Git
```

## Limitations

- Product catalogue data is fictional and intended for retrieval testing.
- Public web results do not establish the shop's actual stock, prices, or policies.
- Each query uses OpenAI and Serper APIs and may incur usage charges.
- The app does not connect to a real order-management, payment, or inventory system.