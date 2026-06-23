# EDUBOT

An intelligent educational assistant that answers questions based on PDF documents. Converts PDFs into chunks, creates embeddings, and uses AI to provide accurate context-based answers.

Converted from a Jupyter notebook into a robust Python package and Streamlit web app.

## Features

- **Multiple AI Providers**: Choose between Google GenAI, Groq, or context-only mode
- **PDF Processing**: Extracts and intelligently chunks PDF content
- **Semantic Search**: Uses embeddings to find relevant context
- **Error Handling**: Robust error handling with helpful user feedback
- **Configuration**: Flexible settings via UI or environment variables
- **Input Validation**: Validates PDFs and questions for optimal results

## Quick Start

### 1. Setup Python Environment

```bash
cd edubot
python -m venv venv
```

Activate the virtual environment:
- **Windows**: `venv\Scripts\Activate.ps1`
- **macOS/Linux**: `source venv/bin/activate`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Keys (Optional)

For AI-powered responses, set one or both API keys:

**Google GenAI:**
```bash
export GENAI_API_KEY="your_google_key_here"
```

**Groq API:**
```bash
export GROQ_API_KEY="your_groq_key_here"
```

Or enter them directly in the app sidebar.

### 4. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Usage

1. Upload a PDF document
2. Enter your question
3. Select an AI provider from the sidebar (or use context-only mode)
4. Click "Generate Answer"

## Configuration Options

- **AI Provider**: Select between No AI, Groq, or Google GenAI
- **Max PDF Size**: 50MB
- **Context Retrieval**: Top 3 most relevant document chunks
- **Embedding Model**: all-MiniLM-L6-v2 (optimized for speed and accuracy)

## Project Structure

- `app.py` - Main Streamlit application
- `pdf_processing.py` - PDF extraction with error handling
- `chunking.py` - Intelligent text chunking
- `embeddings.py` - Embedding generation and FAISS indexing
- `genai_client.py` - Google GenAI and Groq client integration
- `edubot.py` - Core question-answering logic
- `requirements.txt` - Python dependencies

## Error Handling

The application includes comprehensive error handling for:
- Missing or corrupted PDFs
- Empty documents
- Invalid API keys
- Missing dependencies
- Network errors

All errors are displayed in user-friendly messages with helpful tips.
