import fitz
from typing import List


def extract_pages_from_pdf(path: str) -> List[str]:
    """Open a PDF and return a list of page texts (1-based pages).
    
    Args:
        path: Path to the PDF file
        
    Returns:
        List of page texts extracted from the PDF
        
    Raises:
        FileNotFoundError: If the PDF file does not exist
        RuntimeError: If the PDF cannot be opened or is corrupted
    """
    try:
        pdf = fitz.open(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {str(e)}")
    
    if len(pdf) == 0:
        raise RuntimeError("PDF is empty or cannot be read")
    
    pages = []
    for i in range(len(pdf)):
        try:
            page = pdf[i]
            text = page.get_text()
            pages.append(text)
        except Exception as e:
            # Log error but continue processing other pages
            print(f"Warning: Failed to extract page {i+1}: {str(e)}")
            pages.append("")

    pdf.close()
    return pages
