import streamlit as st
import fitz  # PyMuPDF
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage
import os
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import tempfile
import time

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Multimodal RAG Chat Application")
st.markdown("Upload a PDF and chat with both its text and visual content!")

# Initialize session state
if 'processed_pdf' not in st.session_state:
    st.session_state.processed_pdf = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'all_docs' not in st.session_state:
    st.session_state.all_docs = []
if 'image_data_store' not in st.session_state:
    st.session_state.image_data_store = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'clip_model' not in st.session_state:
    st.session_state.clip_model = None
if 'clip_processor' not in st.session_state:
    st.session_state.clip_processor = None

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key input
    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Enter your Google API key for Gemini"
    )
    
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        st.success("✅ API Key set!")
    else:
        st.warning("⚠️ Please enter your Google API key")
    
    st.divider()
    
    # File upload
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF file to analyze"
    )
    
    # Processing parameters
    st.header("⚙️ Parameters")
    chunk_size = st.slider("Text Chunk Size", 200, 1000, 500)
    chunk_overlap = st.slider("Chunk Overlap", 50, 200, 100)
    top_k = st.slider("Retrieved Documents", 3, 10, 5)

@st.cache_resource
def load_clip_model():
    """Load CLIP model and processor"""
    with st.spinner("Loading CLIP model..."):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
    return model, processor

def embed_image(image_data, model, processor):
    """Embed an image using CLIP model."""
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze().numpy()

def embed_text(text, model, processor):
    """Embed text using CLIP model."""
    inputs = processor(
        text=text, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.squeeze().numpy()

class DummyEmbeddings:
    """Dummy embedding class for FAISS"""
    def embed_documents(self, texts):
        return []
    
    def embed_query(self, text):
        return []

def process_pdf(uploaded_file, chunk_size, chunk_overlap):
    """Process the uploaded PDF file"""
    
    # Load CLIP model
    if st.session_state.clip_model is None:
        st.session_state.clip_model, st.session_state.clip_processor = load_clip_model()
    
    model = st.session_state.clip_model
    processor = st.session_state.clip_processor
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Open PDF
        doc = fitz.open(tmp_path)
        
        all_docs = []
        all_embeddings = []
        image_data_store = {}
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_pages = len(doc)
        
        for i, page in enumerate(doc):
            progress = (i + 1) / total_pages
            progress_bar.progress(progress)
            status_text.text(f"Processing page {i+1}/{total_pages}")
            
            # Process text
            text = page.get_text()
            if text.strip():
                temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
                text_chunks = splitter.split_documents([temp_doc])
                
                for chunk in text_chunks:
                    embedding = embed_text(chunk.page_content, model, processor)
                    all_embeddings.append(embedding)
                    all_docs.append(chunk)
            
            # Process images
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    
                    # Create unique identifier
                    image_id = f"page_{i}_img_{img_index}"
                    
                    # Store image as base64
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    image_data_store[image_id] = img_base64
                    
                    # Embed image using CLIP
                    embedding = embed_image(pil_image, model, processor)
                    all_embeddings.append(embedding)
                    
                    # Create document for image
                    image_doc = Document(
                        page_content=f"[Image: {image_id}]",
                        metadata={"page": i, "type": "image", "image_id": image_id}
                    )
                    all_docs.append(image_doc)
                    
                except Exception as e:
                    st.warning(f"Error processing image {img_index} on page {i}: {e}")
                    continue
        
        doc.close()
        
        # Create vector store
        status_text.text("Creating vector store...")
        embeddings_array = np.array(all_embeddings)
        
        vector_store = FAISS.from_embeddings(
            text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
            embedding=DummyEmbeddings(),
            metadatas=[doc.metadata for doc in all_docs]
        )
        
        progress_bar.empty()
        status_text.empty()
        
        return vector_store, all_docs, image_data_store, len(all_docs)
        
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

def retrieve_multimodal(query, vector_store, top_k=5):
    """Retrieve relevant documents and images based on the query."""
    model = st.session_state.clip_model
    processor = st.session_state.clip_processor
    
    query_embedding = embed_text(query, model, processor)
    
    # Retrieve documents
    results = vector_store.similarity_search_by_vector(
        query_embedding,
        k=top_k,
        filter=None
    )
    
    return results

def create_multimodal_message(query, retrieved_docs, image_data_store):
    """Create a message with both text and images for the LLM."""
    content = []
    
    # Add the query
    content.append({
        "type": "text",
        "text": f"Question: {query}\n\nContext:\n"
    })
    
    # Separate text and image documents
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    
    # Add text context
    if text_docs:
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"Text excerpts:\n{text_context}\n"
        })
    
    # Add images
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "text",
                "text": f"\n[Image from page {doc.metadata['page']}]:\n"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data_store[image_id]}"
                }
            })
    
    # Add instruction
    content.append({
        "type": "text",
        "text": "\n\nPlease answer the question based on the provided text and images."
    })
    
    return HumanMessage(content=content)

def multimodal_rag_pipeline(query, vector_store, image_data_store, top_k=5):
    """Main pipeline for multimodal RAG."""
    if not google_api_key:
        return "Please provide a Google API key in the sidebar."
    
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        # Retrieve relevant documents
        context_docs = retrieve_multimodal(query, vector_store, top_k)
        
        # Create multimodal message
        message = create_multimodal_message(query, context_docs, image_data_store)
        
        # Get response from LLM
        response = llm.invoke([message])
        
        return response.content, context_docs
        
    except Exception as e:
        return f"Error: {str(e)}", []

# Main app logic
if uploaded_file is not None:
    if not st.session_state.processed_pdf or uploaded_file.name != getattr(st.session_state, 'last_file_name', None):
        
        if google_api_key:
            with st.spinner("Processing PDF... This may take a few minutes."):
                try:
                    vector_store, all_docs, image_data_store, num_docs = process_pdf(
                        uploaded_file, chunk_size, chunk_overlap
                    )
                    
                    st.session_state.vector_store = vector_store
                    st.session_state.all_docs = all_docs
                    st.session_state.image_data_store = image_data_store
                    st.session_state.processed_pdf = True
                    st.session_state.last_file_name = uploaded_file.name
                    
                    st.success(f"✅ PDF processed successfully! Created {num_docs} document chunks.")
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
        else:
            st.warning("⚠️ Please enter your Google API key first.")

# Chat interface
if st.session_state.processed_pdf and google_api_key:
    
    # Display some statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        text_docs = [doc for doc in st.session_state.all_docs if doc.metadata.get("type") == "text"]
        st.metric("Text Chunks", len(text_docs))
    
    with col2:
        image_docs = [doc for doc in st.session_state.all_docs if doc.metadata.get("type") == "image"]
        st.metric("Images", len(image_docs))
    
    with col3:
        st.metric("Total Documents", len(st.session_state.all_docs))
    
    st.divider()
    
    # Chat interface
    st.header("💬 Chat with your PDF")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if message["role"] == "assistant" and "context" in message:
                with st.expander("📋 Retrieved Context"):
                    for i, doc in enumerate(message["context"]):
                        doc_type = doc.metadata.get("type", "unknown")
                        page = doc.metadata.get("page", "?")
                        
                        if doc_type == "text":
                            st.write(f"**Text from page {page}:**")
                            st.write(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                        else:
                            st.write(f"**Image from page {page}**")
                            if "image_id" in doc.metadata:
                                image_id = doc.metadata["image_id"]
                                if image_id in st.session_state.image_data_store:
                                    image_data = base64.b64decode(st.session_state.image_data_store[image_id])
                                    st.image(image_data, width=200)
                        
                        if i < len(message["context"]) - 1:
                            st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, context_docs = multimodal_rag_pipeline(
                    prompt, 
                    st.session_state.vector_store, 
                    st.session_state.image_data_store, 
                    top_k
                )
            
            st.write(response)
            
            # Show retrieved context
            if context_docs:
                with st.expander("📋 Retrieved Context"):
                    for i, doc in enumerate(context_docs):
                        doc_type = doc.metadata.get("type", "unknown")
                        page = doc.metadata.get("page", "?")
                        
                        if doc_type == "text":
                            st.write(f"**Text from page {page}:**")
                            st.write(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                        else:
                            st.write(f"**Image from page {page}**")
                            if "image_id" in doc.metadata:
                                image_id = doc.metadata["image_id"]
                                if image_id in st.session_state.image_data_store:
                                    image_data = base64.b64decode(st.session_state.image_data_store[image_id])
                                    st.image(image_data, width=200)
                        
                        if i < len(context_docs) - 1:
                            st.divider()
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response,
            "context": context_docs
        })

    # Sample questions
    st.subheader("💡 Sample Questions")
    sample_questions = [
        "What does the chart show about revenue trends?",
        "Summarize the main findings from the document",
        "What visual elements are present in the document?",
        "Explain the data shown in the images",
        "What are the key insights from this document?"
    ]
    
    col1, col2 = st.columns(2)
    for i, question in enumerate(sample_questions):
        col = col1 if i % 2 == 0 else col2
        if col.button(question, key=f"sample_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.rerun()

elif not uploaded_file:
    st.info("👆 Please upload a PDF file to get started!")
    
    # Features section
    st.header("🚀 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📄 Text Processing**
        - Extract and chunk text from PDFs
        - Semantic search across text content
        - Context-aware retrieval
        """)
        
        st.markdown("""
        **🖼️ Image Processing**
        - Extract images from PDFs
        - Visual understanding with CLIP
        - Multimodal search capabilities
        """)
    
    with col2:
        st.markdown("""
        **🤖 AI Chat**
        - Powered by Google Gemini
        - Multimodal responses
        - Context-aware answers
        """)
        
        st.markdown("""
        **⚙️ Customizable**
        - Adjustable chunk sizes
        - Configurable retrieval parameters
        - Interactive interface
        """)

else:
    st.warning("⚠️ Please enter your Google API key to process the PDF.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
Built with ❤️ using Streamlit, LangChain, CLIP, and Google Gemini
</div>
""", unsafe_allow_html=True)
