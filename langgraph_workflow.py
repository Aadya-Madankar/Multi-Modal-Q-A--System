from langgraph.graph import StateGraph
from typing import TypedDict, List, Optional, Any
from PIL import Image
import os
import requests # For requests.exceptions.RequestException in process_url_node
import traceback # For printing traceback in run_graph_workflow

# Import functions from our combined processing file
import document_processing as dp

# --- LangGraph State Definition ---
class GraphState(TypedDict):
    input_type: Optional[str]
    file_path: Optional[str]
    url: Optional[str]
    question: Optional[str]
    document_id: Optional[str] # For PDF query: 'filename_pdf'; For URL query: original URL; For Image query: temp_image_path
    text_content: Optional[List[tuple[str, int]]]
    extracted_pil_images: Optional[List[Image.Image]]
    retrieved_docs: Optional[List[Any]] # Langchain Document
    answer: Optional[str]
    error_message: Optional[str]
# --- End State Definition ---


# --- LangGraph Nodes ---
def start_node(state: GraphState) -> GraphState:
    print("--- Starting Graph Workflow ---")
    current_error = state.get("error_message")
    current_answer = state.get("answer")
    updates = {
        "error_message": current_error,
        "answer": current_answer,
        "extracted_pil_images": state.get("extracted_pil_images", []), # Preserve if passed for some reason
        "text_content": state.get("text_content", []), # Preserve
        "retrieved_docs": [], # Always clear for a new query part of the flow
    }
    # Explicitly set/override with inputs for this run
    updates["input_type"] = state.get("input_type")
    updates["file_path"] = state.get("file_path")
    updates["url"] = state.get("url")
    updates["question"] = state.get("question")
    updates["document_id"] = state.get("document_id")
    return updates

def process_pdf_node(state: GraphState) -> GraphState:
    print("--- Processing PDF ---")
    try:
        file_path = state.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return {"error_message": "PDF file path missing or file not found."}
        
        base_filename = os.path.basename(file_path)
        doc_id = os.path.splitext(base_filename)[0] + "_pdf"
        updates = {"document_id": doc_id} # Set doc_id early

        text_pages = dp.get_pdf_text_and_pages(file_path)
        if not text_pages:
            updates["error_message"] = f"Failed to extract text from PDF: {file_path}"
            return updates
        updates["text_content"] = text_pages
        
        chunks = dp.get_text_chunks(text_pages)
        if not chunks:
            updates["error_message"] = "No text chunks from PDF."
            return updates
            
        if not dp.create_and_save_faiss_index(chunks, doc_id):
            updates["error_message"] = f"Failed to create vector store for PDF: {doc_id}"
        return updates
    except Exception as e:
        print(f"!!! UNHANDLED EXCEPTION in process_pdf_node: {e}")
        return {"error_message": f"Critical error in PDF processing: {str(e)}", 
                "document_id": state.get("document_id") or (os.path.splitext(os.path.basename(state.get("file_path","unknown.pdf")))[0] + "_pdf" if state.get("file_path") else None)}


def process_url_node(state: GraphState) -> GraphState:
    print("--- Processing URL ---")
    try:
        url = state.get("url")
        if not url:
            return {"error_message": "URL not provided."}
        
        query_document_id = url 
        url_faiss_index_name = f"url_{str(hash(url))}_text"
        updates = {"document_id": query_document_id} 
        
        current_error_parts = [] # Collect error messages

        text_data, pil_images = dp.get_url_content_and_images(url)
        
        if pil_images: 
            updates["extracted_pil_images"] = pil_images
        
        if text_data:
            updates["text_content"] = text_data
            chunks = dp.get_text_chunks(text_data)
            if chunks:
                if not dp.create_and_save_faiss_index(chunks, url_faiss_index_name):
                    current_error_parts.append("Failed to index URL text.")
            else: 
                current_error_parts.append("No text chunks from URL.")
        
        if not updates.get("text_content") and not updates.get("extracted_pil_images"):
            current_error_parts.append("No processable content (text or images) from URL.")

        if current_error_parts:
            updates["error_message"] = " ".join(current_error_parts)
            
        return updates
    except requests.exceptions.RequestException as req_e:
        print(f"!!! RequestException in process_url_node for {state.get('url')}: {req_e}")
        return {"error_message": f"Error fetching URL: {str(req_e)}", "document_id": state.get("url")}
    except Exception as e:
        print(f"!!! UNHANDLED EXCEPTION in process_url_node for {state.get('url')}: {e}")
        return {"error_message": f"Critical error in URL processing: {str(e)}", "document_id": state.get("url")}


def process_direct_image_node(state: GraphState) -> GraphState:
    print("--- Processing Direct Image (loader) ---")
    try:
        file_path = state.get("file_path") 
        if not file_path or not os.path.exists(file_path):
            # document_id should be file_path for direct image processing/querying
            return {"error_message": "Image file path missing or file not found.", "document_id": file_path or state.get("document_id")}
        
        img = dp.load_image_from_path(file_path)
        if img:
            return {"extracted_pil_images": [img], "document_id": file_path} # Ensure doc_id is set to file_path
        else:
            return {"error_message": f"Failed to load image: {file_path}", "document_id": file_path}
    except Exception as e:
        print(f"!!! UNHANDLED EXCEPTION in process_direct_image_node: {e}")
        return {"error_message": f"Critical error loading direct image: {str(e)}", "document_id": state.get("file_path") or state.get("document_id")}


def retrieve_text_context_node(state: GraphState) -> GraphState:
    print("--- Retrieving Text Context ---")
    try:
        question = state.get("question")
        user_doc_id = state.get("document_id") 
        if not question or not user_doc_id:
            return {"error_message": "Question or document ID missing for text retrieval."}

        faiss_index_name = ""
        input_type = state.get("input_type")
        if input_type == "pdf":
            faiss_index_name = user_doc_id
        elif input_type == "url":
            faiss_index_name = f"url_{str(hash(user_doc_id))}_text" 
        
        if not faiss_index_name:
            return {"error_message": f"Could not determine FAISS index name for query context '{input_type}' with ID: {user_doc_id}"}

        v_store = dp.load_faiss_index(faiss_index_name)
        if not v_store:
            return {"error_message": f"Could not load document index: {faiss_index_name}"}
        
        retrieved = dp.similarity_search_in_index(v_store, question)
        if not retrieved:
            return {"answer": "No relevant information found in the document for your question.", "retrieved_docs": []}
        return {"retrieved_docs": retrieved}
    except Exception as e:
        print(f"!!! UNHANDLED EXCEPTION in retrieve_text_context_node: {e}")
        return {"error_message": f"Critical error retrieving text context: {str(e)}"}


def generate_text_answer_node(state: GraphState) -> GraphState:
    print("--- Generating Text Answer ---")
    try:
        if state.get("answer"): return {}
            
        question = state.get("question")
        retrieved_docs = state.get("retrieved_docs")
        if not question or not retrieved_docs: # Allow empty retrieved_docs if QA chain can handle (it can't well)
            if not retrieved_docs: # If retriever found nothing, this node should not have been called if "answer" was set
                 return {"error_message": "No documents retrieved to generate text answer."}
            return {"error_message": "Missing data (question or documents) for text answer generation."}
            
        qa_chain = dp.get_conversational_qa_chain()
        if not qa_chain:
            return {"error_message": "Could not initialize QA chain (Likely API key issue)."}
            
        answer_text = dp.query_text_with_llm(qa_chain, retrieved_docs, question)
        return {"answer": answer_text}
    except Exception as e:
        print(f"!!! UNHANDLED EXCEPTION in generate_text_answer_node: {e}")
        return {"error_message": f"Critical error generating text answer: {str(e)}"}


def generate_image_answer_node(state: GraphState) -> GraphState:
    print("--- Generating Image Answer ---")
    try:
        question = state.get("question")
        images = state.get("extracted_pil_images")
        if not question or not images:
            return {"error_message": "Missing data (question or images) for image answer generation."}
            
        answer_text = dp.query_image_with_llm(images, question)
        return {"answer": answer_text}
    except Exception as e:
        print(f"!!! UNHANDLED EXCEPTION in generate_image_answer_node: {e}")
        return {"error_message": f"Critical error generating image answer: {str(e)}"}


def final_result_node(state: GraphState) -> GraphState:
    print("--- Workflow Complete ---")
    if state.get("error_message"): print(f"Final Error: {state['error_message']}")
    if state.get("answer"): print(f"Final Answer: {state.get('answer', '')[:100]}...")
    return {}
# --- End Nodes ---


# --- Conditional Edge Functions ---
def route_initial_processing_or_query(state: GraphState) -> str:
    if state.get("question"): 
        return "query_router_junction_node"
    else: 
        input_type = state.get("input_type")
        if input_type == "pdf": return "process_pdf_node"
        elif input_type == "url": return "process_url_node"
        elif input_type == "image": return "process_direct_image_node"
        else:
            print(f"Error in route_initial: Invalid input_type '{input_type}' for processing.")
            # To set an error message, we'd need an intermediate error-setting node,
            # as conditional functions shouldn't modify state.
            # For now, this will lead to final_result_node.
            return "final_result_node"

def should_proceed_to_query(state: GraphState) -> str:
    if state.get("error_message"): return "final_result_node"
    if state.get("question"): return "query_router_junction_node"
    return "final_result_node"

def decide_query_path(state: GraphState) -> str:
    # This function should only decide based on current state, not set error_message.
    # Errors should be set by the nodes that encounter them.
    if state.get("error_message"): return "final_result_node" # If prior node set error

    query_context_type = state.get("input_type") 
    user_doc_id = state.get("document_id") # This is the ID API uses to refer to content

    if query_context_type == "pdf":
        # For PDF, user_doc_id is the FAISS index name.
        if user_doc_id and os.path.exists(os.path.join(dp.FAISS_INDEX_DIR, user_doc_id)):
            return "retrieve_text_context_node"
        else:
            print(f"Warning in decide_query_path: PDF index '{user_doc_id}' not found for query. Routing to end.")
            return "final_result_node" # Error should have been set by API if doc_id is invalid.

    elif query_context_type == "url":
        # For URL, user_doc_id is the original URL.
        if state.get("extracted_pil_images"): # Prioritize images if loaded in this graph run
            return "generate_image_answer_node"
        
        url_faiss_index_name = f"url_{str(hash(user_doc_id))}_text" if user_doc_id else None
        if url_faiss_index_name and os.path.exists(os.path.join(dp.FAISS_INDEX_DIR, url_faiss_index_name)):
            return "retrieve_text_context_node"
        else:
            print(f"Warning in decide_query_path: For URL query '{user_doc_id}', no images in state and no text index '{url_faiss_index_name}'. Routing to end.")
            return "final_result_node"

    elif query_context_type == "image":
        # For Image, user_doc_id is the file_path.
        if state.get("extracted_pil_images"): # Image already loaded (e.g. /process/image then immediate Q)
            return "generate_image_answer_node"
        elif state.get("file_path") and os.path.exists(state.get("file_path")): # Image needs loading
            print(f"Image query for {state.get('file_path')}, image not in state. Routing to load it.")
            return "process_direct_image_node"
        else:
            print(f"Warning in decide_query_path: Image data for '{user_doc_id}' not found, and no valid file_path. Routing to end.")
            return "final_result_node"
            
    print(f"Warning in decide_query_path: Unknown query context type '{query_context_type}'. Routing to end.")
    return "final_result_node"

def after_text_retrieval_condition(state: GraphState) -> str:
    if state.get("error_message"): return "final_result_node"
    if state.get("answer"): return "final_result_node" # e.g. "No relevant info found"
    if not state.get("retrieved_docs"): # Double check, though 'answer' should be set above
        print("Warning in after_text_retrieval: No docs retrieved but no answer set. Routing to end.")
        return "final_result_node"
    return "generate_text_answer_node"
# --- End Conditional Edge Functions ---


# --- Graph Construction ---
workflow = StateGraph(GraphState)

# Define all nodes
workflow.add_node("start_node", start_node)
workflow.add_node("process_pdf_node", process_pdf_node)
workflow.add_node("process_url_node", process_url_node)
workflow.add_node("process_direct_image_node", process_direct_image_node)

def query_router_junction_node_func(state: GraphState) -> GraphState:
    print("--- At Query Router Junction ---")
    return {} # This node is primarily a routing target
workflow.add_node("query_router_junction_node", query_router_junction_node_func)

workflow.add_node("retrieve_text_context_node", retrieve_text_context_node)
workflow.add_node("generate_text_answer_node", generate_text_answer_node)
workflow.add_node("generate_image_answer_node", generate_image_answer_node)
workflow.add_node("final_result_node", final_result_node)

# Set entry point
workflow.set_entry_point("start_node")

# Conditional Edges from start_node
workflow.add_conditional_edges(
    "start_node",
    route_initial_processing_or_query,
    {
        "process_pdf_node": "process_pdf_node",
        "process_url_node": "process_url_node",
        "process_direct_image_node": "process_direct_image_node",
        "query_router_junction_node": "query_router_junction_node",
        "final_result_node": "final_result_node"
    }
)

# Conditional Edges after processing nodes
workflow.add_conditional_edges("process_pdf_node", should_proceed_to_query, {
    "query_router_junction_node": "query_router_junction_node", "final_result_node": "final_result_node"
})
workflow.add_conditional_edges("process_url_node", should_proceed_to_query, {
    "query_router_junction_node": "query_router_junction_node", "final_result_node": "final_result_node"
})

# Conditional Edges after process_direct_image_node (image loader)
workflow.add_conditional_edges(
    "process_direct_image_node",
    # If a question exists AND image loaded successfully (no error, images in state), then generate answer.
    # Otherwise, it was a standalone process or an error occurred, so end.
    lambda state: "generate_image_answer_node" if state.get("question") and state.get("extracted_pil_images") and not state.get("error_message") else "final_result_node",
    {
        "generate_image_answer_node": "generate_image_answer_node",
        "final_result_node": "final_result_node"
    }
)

# Conditional Edges from the query router junction
workflow.add_conditional_edges(
    "query_router_junction_node",
    decide_query_path,
    {
        "retrieve_text_context_node": "retrieve_text_context_node",
        "generate_image_answer_node": "generate_image_answer_node",
        "process_direct_image_node": "process_direct_image_node", # If query needs image loading
        "final_result_node": "final_result_node"
    }
)

# Conditional Edges after retrieving text context
workflow.add_conditional_edges(
    "retrieve_text_context_node",
    after_text_retrieval_condition,
    {
        "generate_text_answer_node": "generate_text_answer_node",
        "final_result_node": "final_result_node"
    }
)

# Edges from answer generation nodes to the final result node
workflow.add_edge("generate_text_answer_node", "final_result_node")
workflow.add_edge("generate_image_answer_node", "final_result_node")

# Compile the graph
compiled_graph = workflow.compile()

def run_graph_workflow(initial_state_dict: dict) -> dict:
    complete_initial_state = {key: initial_state_dict.get(key) for key in GraphState.__annotations__}
    for key in ["input_type", "file_path", "url", "question", "document_id"]:
        if key not in complete_initial_state:
            complete_initial_state[key] = None # Ensure presence for routing logic
            
    try:
        if 'compiled_graph' not in globals() or compiled_graph is None:
            print("!!! FATAL: compiled_graph is not defined or None in run_graph_workflow.")
            # Ensure error_message is part of the returned dict
            complete_initial_state["error_message"] = "Graph not compiled or available."
            return complete_initial_state

        final_state = compiled_graph.invoke(complete_initial_state)
        
        if final_state is None: # Should ideally not happen if nodes return dicts
            print("!!! Graph execution returned None. This is unexpected.")
            complete_initial_state["error_message"] = "Graph execution failed and returned None."
            return complete_initial_state
        
        # Ensure final_state is a dict and has all keys, merging with initial if some are missing
        # This is important if a node returns only partial updates.
        # LangGraph's invoke should return the full final state, but this is defensive.
        merged_final_state = {**complete_initial_state, **final_state}
        return merged_final_state

    except Exception as e:
        print(f"!!! EXCEPTION during graph invocation: {e}")
        traceback.print_exc()
        complete_initial_state["error_message"] = f"Core graph execution error: {str(e)}"
        return complete_initial_state
# --- End Graph Construction ---