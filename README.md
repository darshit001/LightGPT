# 🧠 LightGPT - Groq LLM Chat Assistant with Custom Tooling

**LightGPT** is an intelligent, extensible chat assistant powered by Groq's LLM and custom tools, built with Streamlit and Starlette. It enables contextual, task-aware responses across multiple domains, including code generation, research, PDF Q&A, image creation, and web search — all through an interactive multi-session chat UI.

## ✨ Features

- ⚙️ **Integrated Groq's LLM** with tool invocation via a custom MCP (Modular Command Processor) server using Server-Sent Events (SSE).
- 🧠 **Tool-Enhanced Intelligence**:
  - `generate_code`: Write code in multiple languages with explanations.
  - `deep_research`: Firecrawl-powered multi-source research.
  - `pdf_qa`: Ask questions about uploaded PDFs using LlamaIndex.
  - `tavily_search`: Summarized real-time web results.
  - `generate_image`: AI image generation via Pollinations.
- 💬 **Multi-session Chat Memory** with PostgreSQL-backed persistence and LangChain memory support.
- 📄 **Export Conversations** to PDF format.
- 🖼️ **File Uploads & Image Previews** integrated into the chat.
- 🧱 **Modern UI/UX** with dynamic chat, customizable input controls, and session management.
- 🔐 **Environment-based Configuration** for easy deployment and API key handling.

## 📁 Project Structure

```
├── main.py                # Streamlit app entrypoint
├── mcp_server_sse.py      # Custom MCP tool server (Starlette + SSE)
├── uploaded_pdfs/         # Folder for PDF uploads
├── image/                 # Folder for generated images
├── database/              # Folder for PDF indexes (LlamaIndex)
├── .env                   # Environment variables
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PostgreSQL
- API Keys:
  - `GROQ_API_KEY`
  - `TAVILY_API_KEY`
  - `FIRECRAWL_API_KEY`

### Installation

```bash
git clone https://github.com/yourusername/lightgpt.git
cd lightgpt
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
SERVER_URL=http://localhost:8000/sse
DB_HOST=localhost
DB_NAME=chatbot_db
DB_USER=postgres
DB_PASSWORD=your_db_password
```

### Run the MCP Server

```bash
python mcp_server_sse.py
```

### Run the Streamlit Frontend

```bash
streamlit run main.py
```

## 🧠 Tools Overview

Each tool is registered with the MCP server and auto-discovered in the frontend:
- **`generate_code`**: Converts natural language into runnable code with explanations.
- **`deep_research`**: Crawls and summarizes web sources deeply.
- **`pdf_qa`**: Answers based on PDF content using vector index.
- **`generate_image`**: Creates images from prompts.
- **`general_qa`, `chat_with_assistant`, `math_solver`, `generate_prompt`**, etc.

## 🧪 Example Use Cases

- Ask: *"Create a Python script to scrape weather data."*
- Upload a PDF and ask: *"Summarize chapter 3."*
- Prompt: *"Generate an image of a futuristic city at night."*

## 📦 Dependencies

- Streamlit
- LangChain
- Groq SDK
- LlamaIndex
- Firecrawl
- Tavily
- psycopg2
- Starlette
- Uvicorn
- dotenv
- ReportLab
