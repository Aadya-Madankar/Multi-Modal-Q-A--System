import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from PIL import Image
import io
from urllib.parse import urljoin
from typing import List, Tuple, Any, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    print("FAISS library not found. Please install it: pip install faiss-cpu")
    FAISS = None

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain # Deprecated
from langchain.chains.combine_documents import create_stuff_documents_chain # New way for "stuff"
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate # ChatPromptTemplate is often preferred
from langchain_core.runnables import RunnablePassthrough # For passing things through in chains
from langchain_core.output_parsers import StrOutputParser # For simple string output

from dotenv import load_dotenv

# --- Configuration --- (Identical to previous)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("🔴 Google API Key not found. Please set GOOGLE_API_KEY in .env")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"🔴 Error configuring Google GenAI: {e}. Is your API key valid?")

EMBEDDING_MODEL_NAME = "models/embedding-001"
CHAT_MODEL_NAME = "gemini-1.5-flash"
FAISS_INDEX_DIR = "FAISS_INDEX"
UPLOAD_DIR = "TEMP_UPLOADS"
MAX_IMAGES_FROM_URL = 5
TEXT_CHUNK_SIZE = 10000
TEXT_CHUNK_OVERLAP = 1000
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
# --- End Configuration ---


# --- Document Parsing Logic --- (Identical to previous)
def get_pdf_text_and_pages(pdf_path: str) -> List[Tuple[str, int]]:
    text_with_pages = []
    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_with_pages.append((f"Page Number: {page_num + 1}\n{page_text}", page_num + 1))
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text_with_pages

def get_url_content_and_images(url: str) -> Tuple[List[Tuple[str, int]], List[Image.Image]]:
    text_with_pseudo_pages = []
    pil_images = []
    page_number = 1
    images_processed_count = 0
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15, stream=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()

        if 'text/html' in content_type:
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
                element.decompose()
            text_content = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
            if text_content:
                sections = text_content.split('\n\n')
                for section in sections:
                    if len(section.strip()) > 50:
                         text_with_pseudo_pages.append((f"URL Section: {page_number}\n{section.strip()}", page_number))
                         page_number +=1
            for img_tag in soup.find_all('img'):
                if images_processed_count >= MAX_IMAGES_FROM_URL: break
                img_src = img_tag.get('src')
                if img_src and not img_src.startswith('data:image'):
                    try:
                        abs_img_url = urljoin(url, img_src)
                        img_response = requests.get(abs_img_url, stream=True, timeout=5)
                        img_response.raise_for_status()
                        if img_response.headers.get('content-type', '').startswith('image/'):
                            pil_img = Image.open(io.BytesIO(img_response.content))
                            pil_images.append(pil_img)
                            images_processed_count += 1
                    except Exception: pass
        elif content_type.startswith('image/'):
            try:
                pil_img = Image.open(io.BytesIO(response.content))
                pil_images.append(pil_img)
            except Exception as e: print(f"Error processing direct image URL {url}: {e}")
    except requests.exceptions.RequestException as e: print(f"Error fetching URL {url}: {e}")
    return text_with_pseudo_pages, pil_images

def load_image_from_path(image_path: str) -> Image.Image | None:
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
# --- End Document Parsing Logic ---


# --- Vector Store Logic --- (Identical to previous)
def get_text_chunks(text_with_pages: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    if not text_with_pages: return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP)
    chunks_with_meta = []
    for text, page_num in text_with_pages:
        split_chunks = text_splitter.split_text(text)
        chunks_with_meta.extend([(chunk, page_num) for chunk in split_chunks])
    return chunks_with_meta

def create_and_save_faiss_index(chunks_with_meta: List[Tuple[str, int]], index_name: str) -> bool:
    if not chunks_with_meta or not FAISS or not GOOGLE_API_KEY:
        if not GOOGLE_API_KEY: print("Cannot create FAISS index: GOOGLE_API_KEY not configured.")
        return False
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
        texts = [chunk[0] for chunk in chunks_with_meta]
        metadatas = [{"source_page": chunk[1]} for chunk in chunks_with_meta]
        vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
        index_path = os.path.join(FAISS_INDEX_DIR, index_name)
        vector_store.save_local(index_path)
        return True
    except Exception as e:
        print(f"Error creating/saving FAISS index {index_name}: {e}")
        return False

def load_faiss_index(index_name: str):
    if not FAISS or not GOOGLE_API_KEY:
        if not GOOGLE_API_KEY: print("Cannot load FAISS index: GOOGLE_API_KEY not configured.")
        return None
    try:
        index_path = os.path.join(FAISS_INDEX_DIR, index_name)
        if not os.path.exists(index_path):
            print(f"Index {index_name} not found at {index_path}")
            return None
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading FAISS index {index_name}: {e}")
        return None

def similarity_search_in_index(vector_store, query: str, k: int = 3) -> List[Any]:
    if not vector_store: return []
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []
# --- End Vector Store Logic ---


# --- LLM Handler Logic (Updated) ---
def get_conversational_qa_chain(): # Renamed, was get_conversational_chain previously
    if not GOOGLE_API_KEY:
        print("Cannot get QA chain: GOOGLE_API_KEY not configured.")
        return None
    
    # Using ChatPromptTemplate for more flexibility if system messages are needed later
    # For simple "stuff" prompt, PromptTemplate is also fine.
    # The system message is implicitly handled by Gemini if you structure the prompt well.
    # If you wanted an explicit system message for Gemini, you'd often prepend it to the user's query.
    # Or use specific message types like SystemMessage, HumanMessage from langchain_core.messages
    
    # Updated prompt for clarity and to use input_variables expected by create_stuff_documents_chain
    # The chain expects "context" (from documents) and "input" (the user's question).
    # If you use a different variable for question (e.g. "question"), you map it when invoking.
    prompt_str = """
    You are an AI assistant. Answer the following question based ONLY on the provided context.
    If the answer is not found in the context, state "The answer is not available in the provided context."
    Do not use any external knowledge. Be concise and accurate.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_str)
    
    # Remove convert_system_message_to_human, as it's deprecated and Gemini handles prompts well.
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL_NAME, 
        temperature=0.3, 
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create the "stuff" documents chain
    # This chain takes 'context' (documents) and 'input' (question)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    return question_answer_chain # This is now a Runnable, not the old Chain type

def query_text_with_llm(qa_chain, documents: List[Any], question: str) -> str:
    if not qa_chain: 
        return "Error: QA chain not available."
    try:
        # The create_stuff_documents_chain expects 'input' for the question
        # and 'context' for the documents.
        # The 'documents' are already Langchain Document objects from FAISS.
        response = qa_chain.invoke({"input": question, "context": documents})
        # The output of create_stuff_documents_chain is typically a string directly.
        if isinstance(response, str):
            return response
        elif isinstance(response, dict) and "answer" in response: # Some chains might wrap
            return response["answer"]
        else:
            print(f"Unexpected response type from QA chain: {type(response)}, content: {response}")
            return "Could not parse answer from QA chain."
            
    except Exception as e:
        print(f"Error querying text with LLM: {e}")
        return f"Error during LLM query: {e}"

def query_image_with_llm(image_content: Union[Image.Image, List[Image.Image]], question: str) -> str:
    if not GOOGLE_API_KEY: return "Error: Image query not available, GOOGLE_API_KEY not configured."
    try:
        # For direct genai SDK usage, model initialization is fine as is.
        model = genai.GenerativeModel(CHAT_MODEL_NAME) 
        prompt_parts = []
        if isinstance(image_content, list): prompt_parts.extend(image_content)
        else: prompt_parts.append(image_content)
        prompt_parts.append(question) # Question is the last part of the multimodal prompt
        
        response = model.generate_content(prompt_parts)

        if response.parts: return "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif hasattr(response, 'text'): return response.text # Older genai versions might have .text directly
        else:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return f"Content analysis blocked: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
            return "Could not get a response from the image model (empty or unexpected structure)."
    except Exception as e:
        print(f"Error querying image with LLM: {e}")
        return f"Error during image analysis: {e}"
# --- End LLM Handler Logic ---