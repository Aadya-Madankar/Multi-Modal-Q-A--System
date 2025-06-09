from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, Any
import uvicorn
import os
import shutil
import uuid

from langgraph_workflow import run_graph_workflow # GraphState not strictly needed here
import document_processing as dp

# --- Pydantic Schemas ---
class ProcessURLRequest(BaseModel):
    url: HttpUrl

class QueryRequest(BaseModel):
    question: str
    context_type: str
    document_id: Optional[str] = None

class ApiResponse(BaseModel):
    message: Optional[str] = None
    data: Optional[Any] = None
    answer: Optional[str] = None
    error: Optional[str] = None
# --- End Schemas ---

app = FastAPI(title="Simplified Multi-Modal RAG API")

@app.on_event("startup")
async def startup_event():
    os.makedirs(dp.UPLOAD_DIR, exist_ok=True)
    if not dp.GOOGLE_API_KEY:
        print("WARNING: GOOGLE_API_KEY not set. API functionality will be limited.")

@app.post("/process/pdf", response_model=ApiResponse)
async def process_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF allowed.")
    
    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(dp.UPLOAD_DIR, temp_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        initial_state = {
            "input_type": "pdf",
            "file_path": file_path,
        }
        final_state = run_graph_workflow(initial_state)

        if not isinstance(final_state, dict): # Very defensive check
            print(f"!!! process_pdf_endpoint: final_state is not a dict: {final_state}")
            return ApiResponse(error="PDF processing failed: Internal error (invalid graph state).")

        if final_state.get("error_message"):
            return ApiResponse(error=str(final_state["error_message"])) # Ensure it's a string
        
        query_doc_id = final_state.get("document_id")
        return ApiResponse(message=f"PDF '{file.filename}' processed.", data={"query_document_id": query_doc_id})

    except Exception as e:
        print(f"!!! EXCEPTION in process_pdf_endpoint: {e}")
        # if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Server error processing PDF: {str(e)}")
    finally:
        if hasattr(file, 'file') and file.file: file.file.close()


@app.post("/process/url", response_model=ApiResponse)
async def process_url_endpoint(request: ProcessURLRequest):
    initial_state = {
        "input_type": "url",
        "url": str(request.url),
    }
    final_state = run_graph_workflow(initial_state)

    # --- CRITICAL DEBUG AND FIX AREA ---
    if not isinstance(final_state, dict):
        print(f"!!! process_url_endpoint: final_state is NOT a dict: {final_state}")
        # This was the most likely cause of the previous NoneType error.
        # run_graph_workflow should now prevent this, but this is a safety net.
        return ApiResponse(error="URL processing failed: Internal error (invalid graph state).")

    if final_state.get("error_message"):
        # If the graph explicitly set an error_message, use it.
        return ApiResponse(error=str(final_state["error_message"])) 
    
    # If we reach here, final_state is a dict and has no "error_message" from the graph.
    # Proceed to extract data, with defaults if keys are missing (though they shouldn't be).
    extracted_images_list = final_state.get("extracted_pil_images", [])
    num_images = len(extracted_images_list) if isinstance(extracted_images_list, list) else 0
    
    query_doc_id = final_state.get("document_id", str(request.url)) # Default to request URL if not in state
    
    return ApiResponse(
        message=f"URL processed. Text indexed (if any). Images found: {num_images}.",
        data={"query_document_id": query_doc_id, 
              "num_images_extracted": num_images}
    )
    # --- END CRITICAL DEBUG AND FIX AREA ---

@app.post("/process/image", response_model=ApiResponse)
async def process_image_endpoint(file: UploadFile = File(...)):
    allowed_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed: PNG, JPG, JPEG, WEBP.")

    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(dp.UPLOAD_DIR, temp_filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        initial_state = {
            "input_type": "image",
            "file_path": file_path,
            "document_id": file_path # For image processing, doc_id is the file_path
        }
        final_state = run_graph_workflow(initial_state)

        if not isinstance(final_state, dict):
            print(f"!!! process_image_endpoint: final_state is not a dict: {final_state}")
            return ApiResponse(error="Image processing failed: Internal error (invalid graph state).")

        if final_state.get("error_message"):
            return ApiResponse(error=str(final_state["error_message"]))

        # query_document_id for an image is its path, which should be set as document_id in the state
        query_doc_id_from_state = final_state.get("document_id", file_path)
        return ApiResponse(
            message=f"Image '{file.filename}' uploaded.",
            data={"query_document_id": query_doc_id_from_state }
        )
    except Exception as e:
        print(f"!!! EXCEPTION in process_image_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Server error processing image: {str(e)}")
    finally:
        if hasattr(file, 'file') and file.file: file.file.close()

@app.post("/query", response_model=ApiResponse)
async def query_endpoint(request: QueryRequest):
    initial_state = {
        "input_type": request.context_type, 
        "question": request.question,
        "document_id": request.document_id, 
    }
    if request.context_type == "image":
        initial_state["file_path"] = request.document_id # For image query, doc_id is the file_path
    if request.context_type == "url":
        initial_state["url"] = request.document_id # For URL query, doc_id is the original URL

    final_state = run_graph_workflow(initial_state)

    if not isinstance(final_state, dict):
        print(f"!!! query_endpoint: final_state is not a dict: {final_state}")
        return ApiResponse(error="Query failed: Internal error (invalid graph state).")

    # Check for error_message first, then answer.
    if final_state.get("error_message"):
        # Return error, but also include answer if graph partially produced one before erroring
        return ApiResponse(error=str(final_state["error_message"]), answer=final_state.get("answer")) 
    
    # If no error, provide the answer. Default if 'answer' key is missing.
    return ApiResponse(answer=final_state.get("answer", "No answer generated by the graph."))

if __name__ == "__main__":
    uvicorn.run("1_fastapi_server:app", host="0.0.0.0", port=8000, reload=True)