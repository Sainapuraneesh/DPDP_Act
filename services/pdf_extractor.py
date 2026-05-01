import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts text from a PDF file provided as bytes.

    Args:
        pdf_bytes: The PDF content as a bytes object.

    Returns:
        A string containing the extracted text from all pages.
    """
    text = ""
    try:
        # Open the PDF from the bytes stream
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError("Failed to process the PDF file. It might be corrupted or in an unsupported format.") from e
    return text