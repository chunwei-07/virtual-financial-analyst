# Financial Report Analyst AI Agent

## Project Overview

This project is an AI-powered agent designed to analyse PDF financial reports. It leverages Large Language Models (LLMs) via the Gemini API, the LangChain framework for agentic capabilities, and PyMuPDF for text extraction. The agent can understand the content of a financial report, answer user queries, generate summaries, detect revenue trends, and highlight key financial metrics in a conversational manner.

### 🛑 This is a personal project and not open for contributions. Feel free to explore, fork, or reuse with credit, but please do not submit pull requests.


## Why I Built This
Back in February 2025 I tried to learn about investment. As a self-learning investor and tech enthusiast, I wanted a more efficient way to analyze company financial reports without manually digging through dozens of pages. That's why I built the Virtual Financial Analyst AI Agent — a smart, conversational tool that can read financial reports, summarize them, extract key metrics, analyze revenue trends, and answer questions in real time.

Powered by **Gemini models** and **LangChain**, this agent leverages AI to bring financial intelligence closer to everyday investors like myself. Whether you're a beginner trying to understand quarterly earnings or a seasoned investor looking for key numbers fast, this tool is designed to make financial analysis more accessible, faster, and even enjoyable.


## Features

*   **PDF Text Extraction:** Extracts text content from uploaded PDF financial reports using PyMuPDF.
*   **Conversational Interface:** Users can interact with the agent through a command-line interface, asking questions about the report.
*   **Contextual Memory:** The agent retains conversation history to understand follow-up questions.
*   **Custom Analytical Tools:**
    *   **Summarization:** Generates concise summaries of the financial report.
    *   **Revenue Trend Analysis:** Identifies and describes revenue trends mentioned in the report.
    *   **Key Financial Metrics Extraction:** Pulls out important financial figures (e.g., Net Income, EPS).
*   **Streaming Responses:** Agent responses are streamed token-by-token for an improved user experience.
*   **Powered by Gemini & LangChain:** Utilizes Google's Gemini models for its analytical capabilities and LangChain for structuring the agent, tools, and memory.

## Tech Stack

*   **Programming Language:** Python 3.X
*   **LLM API:** Google Gemini API (e.g., Gemini 2.0 Flash Exp)
*   **LLM Framework:** LangChain
*   **PDF Parsing:** PyMuPDF (Fitz)
*   **API Key Management:** python-dotenv
*   **Development Environment:** VS Code

## Project Structure
**financial_report_analyzer/** \
├── **.venv/** # Virtual environment \
├── **data/** # PDF Financial Reports are stored here \
├── **src/** # Source code \
│ ├── **init.py** \
│ ├── **agent.py** # Agent logic, tools integration, memory \
│ ├── **pdf_processor.py** # PDF text extraction \
│ ├── **tools.py** # Custom LangChain tools for analysis \
│ └── **utils.py** # Utility functions \
├── **.env** # Stores API keys (Not uploaded for obvious reason, duh) \
├── **main.py** # Main script to run the agent \
└── **README.md** # This file

## Sample Data Used (in data/)
[Meta's 2024 First Quarter Earnings](https://s21.q4cdn.com/399680738/files/doc_financials/2024/q1/Meta-03-31-2024-Exhibit-99-1_FINAL.pdf) 

## Setup and Installation

1.  **Clone the Repository (if applicable, otherwise create project folder):**
    ```bash
    # git clone <repository-url>
    # cd financial_report_analyzer
    ```
    If you don't have a repository, ensure you have the project folder structure as described above.

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv .venv
    ```
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows (Command Prompt/PowerShell):
        ```bash
        .venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    ```txt
    langchain
    langchain-google-genai
    pymupdf
    python-dotenv
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```
    Check **requirements.txt** for full dependencies.

4. **Configure API Key:**
    *   Obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/).
    *   Create a file named `.env` in the root of the project directory.
    *   Add your API key to the `.env` file:
        ```
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
        ```
    *   **Important:** Ensure `.env` is listed in your `.gitignore` file if you are using Git.


## How to Run

1.  **Place PDF Reports:**
    Put the PDF financial reports you want to analyze into the `data/` directory.

2.  **Run the Agent:**
    Open your terminal, make sure your virtual environment is activated, and navigate to the project's root directory. Then run:
    ```bash
    python main.py
    ```

3.  **Interact with the Agent:**
    *   The script will prompt you to enter the filename of the PDF (from the `data/` folder) you wish to analyze.
    *   Once the PDF is processed and the agent is ready, you can start asking questions.
    *   Type `exit` or `quit` to end the session.

    **Example Queries:**
    *   "Can you summarize this report?"
    *   "What were the revenue trends for [Company Name] in [Period]?"
    *   "List the key financial metrics."
    *   "What was the net income?"
    *   (Follow-up) "What factors influenced it?"


## Future Enhancements

*   **OCR Integration:** Implement OCR (e.g., using PaddleOCR) to extract text from images within PDFs (charts, tables as images).
*   **Advanced RAG:** For extremely large documents or more precise information retrieval, integrate a full RAG pipeline with vector stores.
*   **Quantitative Analysis Tools:** Add tools for performing calculations based on extracted data.
*   **Broader Document Support:** Extend to analyze other financial document types (10-Ks, earnings call transcripts).
*   **GUI:** Develop a graphical user interface using Streamlit, Gradio, or a web framework.
