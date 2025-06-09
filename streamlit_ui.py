import streamlit as st
import requests
import os

FASTAPI_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Document Q&A (Simplified)", layout="wide")
st.header("📚🔗🖼️ AI Q&A (Simplified)")

# Initialize session state
if 'query_document_id' not in st.session_state: st.session_state.query_document_id = None
if 'processed_content_type' not in st.session_state: st.session_state.processed_content_type = None
if 'last_status_message' not in st.session_state: st.session_state.last_status_message = None
if 'num_url_images_processed' not in st.session_state: st.session_state.num_url_images_processed = 0

def reset_state():
    st.session_state.query_document_id = None
    st.session_state.processed_content_type = None
    st.session_state.last_status_message = None
    st.session_state.num_url_images_processed = 0

with st.sidebar:
    st.title("📁 Input Options")
    input_mode = st.radio("Choose content type:", ("PDF", "URL", "IMAGE"), key='input_mode_radio', on_change=reset_state)
    st.markdown("---")

    if input_mode == "PDF":
        st.subheader("📄 Process PDF")
        pdf_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")
        if st.button("Submit & Process PDF"):
            if pdf_file:
                with st.spinner("Processing PDF..."):
                    files = {'file': (pdf_file.name, pdf_file, 'application/pdf')}
                    try:
                        response = requests.post(f"{FASTAPI_URL}/process/pdf", files=files, timeout=120)
                        res_data = response.json()
                        if response.status_code == 200 and not res_data.get("error"):
                            st.session_state.last_status_message = res_data.get("message", "PDF processed.")
                            st.session_state.query_document_id = res_data.get("data", {}).get("query_document_id")
                            st.session_state.processed_content_type = "pdf"
                            st.success(f"{st.session_state.last_status_message} Query ID: {st.session_state.query_document_id}")
                        else:
                            st.error(f"API Error (PDF): {res_data.get('error', response.text)}")
                    except requests.exceptions.RequestException as e: st.error(f"Request failed (PDF): {e}")
            else: st.warning("Please upload a PDF file.")

    elif input_mode == "URL":
        st.subheader("🔗 Process URL")
        url_input = st.text_input("Enter URL:", key="url_processor")
        if st.button("Process URL"):
            if url_input:
                with st.spinner("Processing URL..."):
                    try:
                        response = requests.post(f"{FASTAPI_URL}/process/url", json={"url": url_input}, timeout=120)
                        res_data = response.json()
                        if response.status_code == 200 and not res_data.get("error"):
                            st.session_state.last_status_message = res_data.get("message", "URL processed.")
                            st.session_state.query_document_id = res_data.get("data", {}).get("query_document_id") # Original URL
                            st.session_state.num_url_images_processed = res_data.get("data", {}).get("num_images_extracted", 0)
                            st.session_state.processed_content_type = "url"
                            st.success(f"{st.session_state.last_status_message} Query ID: {st.session_state.query_document_id}. Images: {st.session_state.num_url_images_processed}")
                        else:
                            st.error(f"API Error (URL): {res_data.get('error', response.text)}")
                    except requests.exceptions.RequestException as e: st.error(f"Request failed (URL): {e}")
            else: st.warning("Please enter a URL.")

    elif input_mode == "IMAGE":
        st.subheader("🖼️ Upload Image")
        image_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png", "webp"], key="image_uploader")
        if st.button("Submit Image"):
            if image_file:
                with st.spinner("Uploading image..."):
                    files = {'file': (image_file.name, image_file, image_file.type)}
                    try:
                        response = requests.post(f"{FASTAPI_URL}/process/image", files=files, timeout=60)
                        res_data = response.json()
                        if response.status_code == 200 and not res_data.get("error"):
                            st.session_state.last_status_message = res_data.get("message", "Image uploaded.")
                            st.session_state.query_document_id = res_data.get("data", {}).get("query_document_id") # Temp server path
                            st.session_state.processed_content_type = "image"
                            st.success(f"{st.session_state.last_status_message} Query ID (path): {st.session_state.query_document_id}")
                            st.image(image_file, caption="Uploaded Image")
                        else:
                            st.error(f"API Error (Image): {res_data.get('error', response.text)}")
                    except requests.exceptions.RequestException as e: st.error(f"Request failed (Image): {e}")
            else: st.warning("Please upload an image file.")

    st.markdown("---")
    if st.session_state.last_status_message:
        st.sidebar.info(f"Status: {st.session_state.last_status_message}")


# --- Main Panel for Q&A ---
st.subheader("💬 Ask a Question")

if not st.session_state.query_document_id or not st.session_state.processed_content_type:
    st.info("☝️ Process a document using the sidebar to enable querying.")
else:
    st.success(f"Ready to query: {st.session_state.processed_content_type.upper()} (ID: {st.session_state.query_document_id})")
    if st.session_state.processed_content_type == "url" and st.session_state.num_url_images_processed > 0:
        st.caption(f"This URL has {st.session_state.num_url_images_processed} image(s). Queries might use image content if relevant.")

user_question = st.text_input("Enter your question:", key="user_question_input", \
    disabled=(not st.session_state.query_document_id))

if st.button("Submit Query", disabled=(not st.session_state.query_document_id or not user_question)):
    with st.spinner("Asking API..."):
        payload = {
            "question": user_question,
            "context_type": st.session_state.processed_content_type,
            "document_id": st.session_state.query_document_id
        }
        try:
            response = requests.post(f"{FASTAPI_URL}/query", json=payload, timeout=180)
            res_data = response.json()
            st.subheader("💡 Answer:")
            if response.status_code == 200 and not res_data.get("error"):
                st.markdown(res_data.get("answer", "No answer text received."))
            else:
                st.error(f"API Error (Query): {res_data.get('error', 'Failed to get answer.')}")
                if res_data.get("answer"): # Show partial answer if API provided it alongside error
                    st.markdown(f"*Partial/Previous Answer Attempt:*\n{res_data.get('answer')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed (Query): {e}")
elif st.button("Clear Processed Content & Reset", type="secondary"):
    reset_state()
    st.rerun()