from typing import List, Dict
import re
import logging
import spacy
import time

from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
    EntityRecognizer,
    RecognizerResult,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.operators import Operator, OperatorType

from transformers import pipeline

logger = logging.getLogger(__name__)

# ===================== 1. Custom Anonymizer =====================

class CustomPiiAnonymizer(Operator):
    """
    A custom Presidio operator that replaces PII values based on specific rules.
    """
    def operate(self, text: str, params: dict) -> str:
        entity_type = params.get("entity_type", "PII")

        if entity_type == "PHONE_NUMBER":
            return "[Ph number]"

        simple_mappings = {
            "PERSON": "[Person]",
            "LOCATION": "[Location]",
            "ORGANIZATION": "[Organization]",
            "DATE": "[Date]",
            "TIME": "[Time]",
            "DATE_TIME": "[Date_Time]",
            "EMAIL": "[EMAIL_ADDRESS]",
        }
        if entity_type in simple_mappings:
            return simple_mappings[entity_type]

        # Generic rule for any ID-like entity
        cleaned_text = re.sub(r'[\s-]', '', text.strip())
        if re.fullmatch(r'(?=[a-zA-Z0-9]*[0-9])[a-zA-Z0-9]{6,}', cleaned_text):
            return "[id]"

        # Fallback for any other entity type
        return f"[{entity_type}]"

    def validate(self, params: dict) -> None: pass
    def operator_name(self) -> str: return "custom_pii_rules"
    def operator_type(self) -> OperatorType: return OperatorType.Anonymize


# ===================== 2. Text Chunking Utility =====================

_spacy_nlp = None

def _load_spacy():
    """Loads a small spaCy model for sentence splitting."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        except OSError:
            logger.warning("'en_core_web_sm' not found for chunking. Downloading...")
            from spacy.cli import download
            download("en_core_web_sm")
            _spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    return _spacy_nlp

def _chunk_text(text: str, max_chars: int = 450) -> list[tuple[str, int]]:
    """Splits text into chunks using spaCy's sentence tokenizer for accuracy."""
    nlp = _load_spacy()
    doc = nlp(text)

    chunks = []
    current_chunk = ""
    current_start = 0

    for sent in doc.sents:
        if not current_chunk:
            current_start = sent.start_char

        if len(current_chunk) + len(sent.text_with_ws) > max_chars:
            chunks.append((current_chunk, current_start))
            current_chunk = sent.text_with_ws
            current_start = sent.start_char
        else:
            current_chunk += sent.text_with_ws

    if current_chunk:
        chunks.append((current_chunk, current_start))

    return chunks


# ===================== 3. Custom Recognizers =====================

class GlinerRecognizer(EntityRecognizer):
    """Custom Presidio recognizer for the GLiNER model."""
    def __init__(self, **kwargs):
        self.model_name = "urchade/gliner_multi_pii-v1"
        self.pipeline = None
        self.gliner_labels = ["person", "organization", "email", "phone number"]
        self.label_map = {
            "person": "PERSON", "organization": "ORGANIZATION",
            "email": "EMAIL", "phone number": "PHONE_NUMBER"
        }
        super().__init__(supported_entities=list(self.label_map.values()), **kwargs)

    def load(self) -> None:
        from gliner import GLiNER
        self.pipeline = GLiNER.from_pretrained(self.model_name)

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        if not self.pipeline:
            return []

        results = []
        chunks = _chunk_text(text)
        for chunk, offset in chunks:
            preds = self.pipeline.predict_entities(chunk, self.gliner_labels, threshold=0.5)
            for r in preds:
                entity_type = self.label_map.get(r["label"])
                if entity_type and entity_type in entities:
                    results.append(RecognizerResult(
                        entity_type=entity_type,
                        start=r["start"] + offset,
                        end=r["end"] + offset,
                        score=r["score"]
                    ))
        return results


class AddressNerRecognizer(EntityRecognizer):
    """Custom Presidio recognizer for the Shiprocket Indian Address NER model."""
    def __init__(self, **kwargs):
        self.pipeline = None
        super().__init__(supported_entities=["LOCATION"], **kwargs)

    def load(self) -> None:
        self.pipeline = pipeline("token-classification", model="shiprocket-ai/open-indicbert-indian-address-ner", aggregation_strategy="simple")

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        if not self.pipeline:
            return []

        results = []
        chunks = _chunk_text(text)
        for chunk, offset in chunks:
            preds = self.pipeline(chunk)
            for r in preds:
                if r["score"] > 0.85 and "LOCATION" in entities:
                    results.append(RecognizerResult(
                        entity_type="LOCATION",
                        start=r["start"] + offset,
                        end=r["end"] + offset,
                        score=r["score"]
                    ))
        return results


class ClinicalRecognizer(EntityRecognizer):
    """Custom recognizer for BioMedical NER Model to identify medical terms."""
    def __init__(self, **kwargs):
        self.pipeline = None
        super().__init__(supported_entities=["MEDICAL"], **kwargs)

    def load(self) -> None:
        self.pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        if not self.pipeline:
            return []

        results = []
        chunks = _chunk_text(text)
        for chunk, offset in chunks:
            preds = self.pipeline(chunk)
            for r in preds:
                text_value = chunk[r["start"]:r["end"]]
                if len(text_value.strip()) < 3:
                    continue
                if not any(c.isalpha() for c in text_value):
                    continue

                if r["score"] > 0.7 and "MEDICAL" in entities:
                    label = r.get("entity_group", "UNKNOWN").replace("B-", "").replace("I-", "")
                    results.append(RecognizerResult(
                        entity_type=f"MEDICAL_{label}",
                        start=r["start"] + offset,
                        end=r["end"] + offset,
                        score=r["score"]
                    ))
        return results


# ===================== 4. Presidio Engine Setup =====================

_analyzer = None
_anonymizer = None

def load_presidio_engine():
    """Initializes and configures the Presidio Analyzer and Anonymizer engines."""
    global _analyzer, _anonymizer

    if _analyzer is None:
        registry = RecognizerRegistry()

        # Add our powerful custom models first
        registry.add_recognizer(ClinicalRecognizer())
        registry.add_recognizer(GlinerRecognizer())
        registry.add_recognizer(AddressNerRecognizer())

        # --- Add back high-confidence custom regex recognizers ---

        # For specific, structured IDs like Aadhaar, PAN, and ABHA
        structured_id_recognizer = PatternRecognizer(
            supported_entity="ID",
            name="Structured ID Recognizer",
            patterns=[
                Pattern(name="aadhaar", regex=r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', score=0.95),
                Pattern(name="pan_card", regex=r'\b[A-Z]{5}\d{4}[A-Z]\b', score=0.95),
                Pattern(name="abha_id", regex=r'\b\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', score=0.95),
            ]
        )
        registry.add_recognizer(structured_id_recognizer)

        # For general alphanumeric or numeric IDs with at least 6 characters
        generic_id_recognizer = PatternRecognizer(
            supported_entity="ID", name="Generic ID Recognizer",
            patterns=[
                Pattern(name="alphanumeric_id", regex=r'\b(?=[a-zA-Z0-9]*[a-zA-Z])(?=[a-zA-Z0-9]*[0-9])[a-zA-Z0-9]{6,}\b', score=0.7),
                Pattern(name="numeric_id", regex=r'\b\d{6,}\b', score=0.6)
            ]
        )
        registry.add_recognizer(generic_id_recognizer)

        # For Doctor names, labeled as 'PERSON'
        doctor_recognizer = PatternRecognizer(
            supported_entity="PERSON",
            name="Doctor Name Recognizer",
            patterns=[
                # This pattern looks for "Dr." (case-insensitive) followed by a capitalized name
                # of 1 to 4 words. This is more specific than a generic name pattern.
                Pattern(
                    name="doctor_name_pattern",
                    regex=r'(?i)\bDr\.?\s+(?-i)([A-Z][a-zA-Z]*\.?(?:\s+[A-Z][a-zA-Z]*\.?){0,3})',
                    score=0.9
                ),
            ]
        )
        registry.add_recognizer(doctor_recognizer)

        # For various date formats, labeled as 'DATE'
        date_recognizer = PatternRecognizer(
            supported_entity="DATE",
            name="Custom Date Recognizer",
            patterns=[
                # 1. Big-Endian (ISO): YYYY-MM-DD or YYYY/MM/DD
                Pattern(name="date_iso_8601", regex=r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', score=0.85),
                # 2. Little/Middle-Endian: DD/MM/YYYY, DD-MM-YY, MM/DD/YYYY
                Pattern(name="date_dmy_mdy", regex=r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', score=0.7),
                # 3. Written out dates (e.g., April 30, 2026 or 30th April 2026)
                Pattern(name="date_words_1", regex=r'(?i)\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}\b', score=0.9),
                Pattern(name="date_words_2", regex=r'(?i)\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s+\d{4}\b', score=0.9),
                # 4. Month and Date only (e.g., April 30 or 30th April)
                Pattern(name="month_date", regex=r'(?i)\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b', score=0.75),
                Pattern(name="date_month", regex=r'(?i)\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b', score=0.75),
                # 5. Month and Year only (e.g., April 2026) - Suggested Addition
                Pattern(name="month_year", regex=r'(?i)\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s+\d{4}\b', score=0.8),
                # 6. Month only (e.g., June or AUG)
                Pattern(name="month_only", regex=r'(?i)\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b', score=0.6),
                # 7. Weekday only (e.g., Thursday or Thu)
                Pattern(name="weekday_only", regex=r'(?i)\b(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)\b', score=0.6),
            ]
        )
        registry.add_recognizer(date_recognizer)

        # For various time formats, labeled as 'TIME'
        time_recognizer = PatternRecognizer(
            supported_entity="TIME",
            name="Custom Time Recognizer",
            patterns=[
                # 1. 12-Hour format (e.g., 10:30 PM, 10.30am)
                Pattern(name="time_12_hour", regex=r'(?i)\b(1[0-2]|0?[1-9])[:.][0-5]\d\s*(?:a\.?m\.?|p\.?m\.?)\b', score=0.8),
                # 2. 24-Hour format (e.g., 14:20, 09:15:30)
                Pattern(name="time_24_hour", regex=r'\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b', score=0.75),
            ]
        )
        registry.add_recognizer(time_recognizer)

        # For specific address components, labeled as 'LOCATION'
        address_component_recognizer = PatternRecognizer(
            supported_entity="LOCATION",
            name="Address Component Recognizer",
            patterns=[
                # 1. Flat/Plot numbers (e.g., Flat 402, Plot no: 21)
                Pattern(name="flat_plot_no", regex=r'(?i)\b(?:Flat|Plot)\s*(?:no\.?\s*[:.]?)?\s*[\w-]+', score=0.8),
                # 2. Cross/Main roads (e.g., 12th Cross, 3rd Main road)
                Pattern(name="cross_main_road", regex=r'(?i)\b\d+(?:st|nd|rd|th)?\s+(?:Cross|Main)(?:\s+road)?\b', score=0.8),
                # 3. Floor numbers (e.g., 4th Floor, ground floor)
                Pattern(name="floor_no", regex=r'(?i)\b(?:(?:\d+(?:st|nd|rd|th)?)|Grnd|ground)\s+floor\b', score=0.8),
                # 4. Sector numbers (e.g., Sector 2)
                Pattern(name="sector_no", regex=r'(?i)\bSector\s+\d+\b', score=0.8),
                # 5. House/Flat numbers (e.g., House No: 121, H.No. 4-12)
                Pattern(name="house_flat_no", regex=r'(?i)\b(?:House|H|Flat)\.?\s*No\.?\s*[:\-]?\s*[\w-]+', score=0.8),
                # 6. Site/Shop/Door numbers (e.g., Site No 5, Door 12)
                Pattern(name="site_shop_door_no", regex=r'(?i)\b(?:Site|Shop|Door)\s*(?:No\.?)?\s*[:\-]?\s*\d+\b', score=0.8),
            ]
        )
        registry.add_recognizer(address_component_recognizer)

        # For email addresses
        email_recognizer = PatternRecognizer(
            supported_entity="EMAIL",
            name="Email Recognizer (Regex)",
            patterns=[
                Pattern(
                    name="email_pattern",
                    regex=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    score=0.85
                )
            ]
        )
        registry.add_recognizer(email_recognizer)

        indian_phone_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER", name="Indian Phone Recognizer",
            patterns=[Pattern(name="indian_phone_10_digit", regex=r'\b(?:(?:\+91[\-\s]?)?[6-9]\d{9})\b', score=0.9)]
        )
        registry.add_recognizer(indian_phone_recognizer)

        # --- Task 1: Disable all of Presidio's default recognizers ---
        # We are NOT calling `registry.load_predefined_recognizers()`.
        # This ensures that only our custom models (Clinical, GLiNER, Shiprocket)
        # are used, preventing unexpected detections like UK_NHS or CREDIT_CARD.

        _analyzer = AnalyzerEngine(registry=registry)

    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
        _anonymizer.add_anonymizer(CustomPiiAnonymizer)

    return _analyzer, _anonymizer


# ===================== 5. Main Analysis Function =====================

def analyze_and_anonymize(text: str):
    """
    Analyzes text with a multi-stage filtering process to accurately detect and anonymize PII.
    """
    analyzer, anonymizer = load_presidio_engine()

    start = time.time()
    results = analyzer.analyze(text=text, language="en")
    print("Analyzer Time:", time.time() - start)

    # --- Stage 1: Medical Term Blacklisting (Task 2) ---
    # First, identify all medical terms to create a "blacklist" of spans.
    medical_entities_debug = []
    medical_spans = []
    for r in results:
        if r.entity_type.startswith("MEDICAL"):
            value = text[r.start:r.end]
            medical_spans.append((r.start, r.end))
            if value not in medical_entities_debug:
                medical_entities_debug.append(value)

    def overlap(a, b, c, d):
        return a < d and b > c

    # Filter out any PII that falls within a medical term's span.
    pii_after_medical_filter = []
    for r in results:
        if r.entity_type.startswith("MEDICAL"): continue
        is_medical = any(overlap(r.start, r.end, m_start, m_end) for m_start, m_end in medical_spans)
        if not is_medical:
            pii_after_medical_filter.append(r)

    # --- Stage 2: GLiNER Prioritization (Task 4) ---
    # Create a list of spans for all PII detected by GLiNER.
    gliner_spans = [(r.start, r.end) for r in pii_after_medical_filter if r.recognition_metadata.get("recognizer_name") == "GlinerRecognizer"]

    # Filter out Shiprocket results that overlap with GLiNER results.
    final_pii_results = []
    for r in pii_after_medical_filter:
        if r.recognition_metadata.get("recognizer_name") == "AddressNerRecognizer":
            is_gliner_conflict = any(overlap(r.start, r.end, g_start, g_end) for g_start, g_end in gliner_spans)
            if not is_gliner_conflict:
                final_pii_results.append(r)
        else:
            final_pii_results.append(r)

    # --- Anonymization ---
    # Use the final, filtered list of PII for masking.
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=final_pii_results,
        operators={"DEFAULT": OperatorConfig("custom_pii_rules")}
    )
    safe_text = anonymized.text

    # --- Result Grouping ---
    pii_groups = {}
    pii_by_model = {}
    model_map = {
        "GlinerRecognizer": "GLiNER",
        "AddressNerRecognizer": "Shiprocket",
        "ClinicalRecognizer": "ClinicalBERT"
    }

    # Group the final PII results for the API response.
    for r in final_pii_results:
        value = text[r.start:r.end]
        entity = r.entity_type
        source = r.recognition_metadata.get("recognizer_name", "Unknown")

        # Group by entity type
        pii_groups.setdefault(entity, set()).add(value)

        # Group by source model
        model_name = model_map.get(source, "Regex")
        pii_by_model.setdefault(model_name, {}).setdefault(entity, set()).add(value)

    # Convert sets to sorted lists for consistent JSON output
    for entity_type in pii_groups:
        pii_groups[entity_type] = sorted(list(pii_groups[entity_type]))
    for model_name in pii_by_model:
        for entity_type in pii_by_model[model_name]:
            pii_by_model[model_name][entity_type] = sorted(list(pii_by_model[model_name][entity_type]))

    return safe_text, pii_groups, pii_by_model, medical_entities_debug
