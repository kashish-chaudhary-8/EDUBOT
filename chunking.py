from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict


def chunk_pages(pages: List[str], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict]:
    """Split each page text into chunks and return list of dicts with page metadata.
    
    Args:
        pages: List of page texts to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of dictionaries with 'page' and 'text' keys
        
    Raises:
        ValueError: If pages list is empty or chunk parameters are invalid
    """
    if not pages:
        raise ValueError("Pages list cannot be empty")
    
    if chunk_size <= 0 or chunk_overlap < 0:
        raise ValueError("chunk_size must be positive and chunk_overlap must be non-negative")
    
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked = []

    for i, text in enumerate(pages, start=1):
        if text.strip():  # Only process non-empty pages
            chunks = splitter.split_text(text)
            for c in chunks:
                chunked.append({"page": i, "text": c})

    return chunked
