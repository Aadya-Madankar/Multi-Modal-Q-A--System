# Multi-Modal-Q & A -System

This project is a simplified multi-modal Question & Answering system built with Python. It allows users to upload PDF documents, process content from URLs (text and images), or upload individual images, and then ask questions based on the processed content.

The system leverages a modern stack including FastAPI for the backend API, LangGraph for workflow orchestration, Langchain for core LLM and document processing functionalities, and CrewAI for agent-based task execution (for some processing and Q&A tasks).

## Features

*   **PDF Processing:** Upload PDF files, extract text, create embeddings, and store in a FAISS vector store for semantic search.
*   **URL Processing:**
    *   Extract primary text content from web pages, create embeddings, and store in FAISS.
    *   Extract images from web pages.
*   **Direct Image Upload:** Upload individual image files.
*   **Question Answering:**
    *   Ask questions about processed PDF text (RAG-based).
    *   Ask questions about processed URL text (RAG-based).
    *   Ask questions about images (from URLs or direct uploads) using a multi-modal LLM (Gemini).
*   **Modular Design:** Separated components for API, workflow, document processing, and UI.
*   **Streamlit UI:** A simple web interface for interacting with the system.

## Tech Stack

*   **Backend API:** FastAPI
*   **Workflow Orchestration:** LangGraph
*   **LLM & Document Toolkit:** Langchain
    *   Embeddings: Google Generative AI Embeddings
    *   LLM: Google Gemini (e.g., `gemini-1.5-flash`)
    *   Vector Store: FAISS
*   **Agent Framework:** CrewAI (for selected tasks)
*   **Frontend UI:** Streamlit
*   **Programming Language:** Python

## Setup and Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate it:
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    *   Create a file named `.env` in the `simplified_rag_app` directory.
    *   Add your Google API Key to it:
        ```env
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
        ```
    *   Replace `"YOUR_GOOGLE_API_KEY_HERE"` with your actual API key from Google AI Studio or Google Cloud.

## Running the Application

You need to run two components separately: the FastAPI backend and the Streamlit UI.

1.  **Start the FastAPI Backend Server:**
    Open a terminal in the `simplified_rag_app` directory and run:
    ```bash
    python fastapi_server.py
    ```
    The API server will typically start on `http://localhost:8000`. You should see log messages indicating it has started.

2.  **Start the Streamlit UI:**
    Open another terminal in the `simplified_rag_app` directory and run:
    ```bash
    streamlit run streamlit_ui.py
    ```
    This will open the application in your web browser, usually at `http://localhost:8501`.

## How to Use

1.  Once both the backend and frontend are running, open the Streamlit UI in your browser.
2.  Use the sidebar to select the content type you want to process:
    *   **PDF:** Upload a PDF file and click "Submit & Process PDF".
    *   **URL:** Enter a web URL and click "Process URL".
    *   **IMAGE:** Upload an image file and click "Submit Image".
3.  After successful processing, a "query document ID" will be displayed.
4.  Enter your question about the processed content in the main panel and click "Submit Query".
5.  The answer from the AI will be displayed.

## Testing FastAPI Endpoints (Optional)

You can test the FastAPI endpoints directly using:

*   **Swagger UI:** Navigate to `http://localhost:8000/docs` in your browser once the FastAPI server is running.
*   **ReDoc:** Navigate to `http://localhost:8000/redoc`.
*   Tools like Postman, Insomnia, or `curl`.

## License

(Specify a license, e.g., MIT License, Apache 2.0. If unsure, you can omit this or add "To be determined.")

---
