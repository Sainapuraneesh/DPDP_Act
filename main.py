import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from .routes.report import router as advanced_router
from .services.presidio_stack_service import load_presidio_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the Presidio engine and all custom models on startup."""
    print("Pre-loading Advanced Presidio Stack (GLiNER, AddressNER)...")
    t = time.perf_counter()
    load_presidio_engine()
    print(f"Advanced Presidio Stack ready in {time.perf_counter() - t:.2f}s")
    yield

app = FastAPI(
    title="PII Anonymization Service v7 (Presidio + GLiNER + AddressNER)",
    description="PII detection using Presidio to orchestrate GLiNER, a custom Indian Address NER, and Regex.",
    version="7.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def log_request_time(request: Request, call_next):
    """Middleware to log the processing time of each request."""
    start_time = time.perf_counter()
    response = await call_next(request)
    time_taken = time.perf_counter() - start_time
    print(f'[GLiNER Service] "{request.method} {request.url.path}" {response.status_code} - Completed in {time_taken:.3f}s')
    return response

app.include_router(advanced_router, tags=["PII Detection (Advanced Stack)"])

@app.get("/")
def read_root():
    return {
        "message": "PII Anonymization Service v7",
        "framework": "Presidio + GLiNER + AddressNER + Regex",
        "endpoints": ["/process-report-advanced"]
    }