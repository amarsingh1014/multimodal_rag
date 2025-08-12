# Multimodal RAG Chat Application

A powerful Streamlit application that allows you to chat with PDF documents using both text and visual content. Built with CLIP for multimodal embeddings and Google Gemini for intelligent responses.

## Features

- 📄 **Text Processing**: Extract and chunk text from PDFs with semantic search
- 🖼️ **Image Processing**: Extract and understand images using CLIP embeddings
- 🤖 **AI Chat**: Powered by Google Gemini for multimodal responses
- ⚙️ **Customizable**: Adjustable parameters for chunk sizes and retrieval
- 💬 **Interactive**: Real-time chat interface with context display

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/amarsingh1014/multimodal_rag/blob/main/README.md
cd Multimodal_RAG
```

2. **Install the required dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env and add your Google API key
```

## Usage

1. **Get a Google API Key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key for Gemini

2. **Run the Application**:
```bash
streamlit run app.py
```

3. **Using the App**:
   - Enter your Google API key in the sidebar (or set it in .env file)
   - Upload a PDF file
   - Wait for processing (may take a few minutes for large PDFs)
   - Start chatting with your document!

## How It Works

1. **PDF Processing**: The app extracts both text and images from uploaded PDFs
2. **Text Chunking**: Text is split into manageable chunks using LangChain's text splitter
3. **Multimodal Embeddings**: Both text and images are embedded using OpenAI's CLIP model
4. **Vector Search**: FAISS is used for efficient similarity search across embeddings
5. **Multimodal RAG**: Retrieved context (text + images) is sent to Google Gemini for intelligent responses

## Configuration Options

- **Text Chunk Size**: Control how text is split (200-1000 characters)
- **Chunk Overlap**: Set overlap between chunks (50-200 characters)
- **Retrieved Documents**: Number of relevant documents to retrieve (3-10)

## Sample Questions

Try asking questions like:
- "What does the chart show about revenue trends?"
- "Summarize the main findings from the document"
- "What visual elements are present in the document?"
- "Explain the data shown in the images"

## Technical Details

- **Frontend**: Streamlit for the web interface
- **Embeddings**: OpenAI CLIP for multimodal text/image embeddings
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: Google Gemini 2.5 Flash for response generation
- **PDF Processing**: PyMuPDF for text and image extraction

## Requirements

- Python 3.8+
- Google API key for Gemini
- Internet connection for downloading CLIP models

## Troubleshooting

1. **API Key Issues**: Make sure your Google API key is valid and has access to Gemini
2. **Memory Issues**: For large PDFs, try reducing chunk size or processing smaller files
3. **Model Loading**: CLIP models are downloaded on first use and cached locally
