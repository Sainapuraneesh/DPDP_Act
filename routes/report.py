import time
import logging
from fastapi import APIRouter, HTTPException, Body, Request, File, UploadFile
from pydantic import BaseModel, Field
from typing import Dict, List

from ..services.presidio_stack_service import analyze_and_anonymize
from ..services.pdf_extractor import extract_text_from_pdf

# Configure logging to be less verbose for third-party libraries by default.
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:     %(message)s')
# Get a logger for this specific module.
logger = logging.getLogger(__name__)
# Set this module's logger back to INFO to allow the 200 OK log to appear.
logger.setLevel(logging.INFO)


router = APIRouter()

class ReportRequest(BaseModel):
    text: str = Field(..., description="The raw text content of the report to be processed.")

# class PIIResponse(BaseModel):
#     safe_text: str = Field(..., description="The original text with PII values replaced by placeholders.")
#     pii_groups: Dict[str, List[str]] = Field(..., description="A dictionary mapping PII categories to a list of detected values.")
# ADDED NOW
class PIIResponse(BaseModel):
    safe_text: str = Field(..., description="The anonymized text.")
    pii_groups: Dict[str, List[str]] = Field(..., description="Grouped PII entities.")
    pii_by_model: Dict[str, Dict[str, List[str]]] = Field(
        ..., description="PII grouped by detection model (GLiNER, Regex, Shiprocket)."
    )
    medical_debug: List[str] = Field(
        [], description="List of detected medical terms for debugging purposes."
    )

@router.post("/process-report-advanced", response_model=PIIResponse, summary="Detect and anonymize PII from raw text")
async def process_report(req: ReportRequest = Body(...), http_request: Request = None):
    """
    Accepts raw text and returns detected PII entities and an anonymized version of the text.
    This endpoint uses Presidio to orchestrate:
    1. **GLiNER** for flexible entities (PERSON, etc.).
    2. A fine-tuned **IndicBERT** for Indian addresses (LOCATION).
    3. **Regex** for structured data (IDs, Phones).
    """
    if not req.text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        start_time = time.perf_counter()
        # safe_text, pii_groups, _ = analyze_and_anonymize(req.text)
        # ADDED NOW
        safe_text, pii_groups, pii_by_model, medical_debug = analyze_and_anonymize(req.text)
        time_taken = time.perf_counter() - start_time

        # Manually log the successful request in a format similar to Uvicorn's access logs.
        client_host = http_request.client.host if http_request else "unknown"
        log_message = f'{client_host} - "POST /process-report-advanced HTTP/1.1" 200 OK (Completed in {time_taken:.4f}s)'
        logger.info(log_message)

        # return {"safe_text": safe_text, "pii_groups": pii_groups}
        # ADDED NOW
        return {"safe_text": safe_text, "pii_groups": pii_groups,"pii_by_model": pii_by_model,"medical_debug": medical_debug  }
    except Exception as e:
        logger.error(f"Error during advanced stack processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during PII processing.")

@router.post("/process-pdf-advanced", response_model=PIIResponse, summary="Upload PDF, extract text, and anonymize PII")
async def process_pdf_report(http_request: Request, file: UploadFile = File(...)):
    """
    Accepts a PDF file, extracts its text content, and returns detected PII
    entities and an anonymized version of the text.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        start_time = time.perf_counter()

        pdf_bytes = await file.read()
        extracted_text = extract_text_from_pdf(pdf_bytes)

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the PDF.")

        safe_text, pii_groups, pii_by_model, medical_debug = analyze_and_anonymize(extracted_text)
        time_taken = time.perf_counter() - start_time

        client_host = http_request.client.host if http_request else "unknown"
        log_message = f'{client_host} - "POST /process-pdf-advanced HTTP/1.1" 200 OK (Completed in {time_taken:.4f}s)'
        logger.info(log_message)

        return {"safe_text": safe_text, "pii_groups": pii_groups, "pii_by_model": pii_by_model, "medical_debug": medical_debug}
    except ValueError as ve:
        logger.error(f"Error processing PDF: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during PDF processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during PDF processing.")