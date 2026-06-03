import logging
import json

# import the json recovery function
from .medgemma_response_json_recoverer import attempt_json_recovery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_clinical_output(raw_response: str) -> dict:
    """Parses and validates the raw JSON response from Gemma 4.

    Args:
        raw_response (str): The raw string output from the model

    Returns:
        dict: Validated clinical profile, or a partial result with error flag
    """
    cleaned = raw_response.strip()

    # ── Strip MedGemma extended-thinking tokens ──
    # Model wraps chain-of-thought in <unused94>thought ... <unused95>
    # The actual JSON comes after <unused95>
    if "<unused95>" in cleaned:
        cleaned = cleaned.split("<unused95>", 1)[1].strip()
    elif "<unused94>" in cleaned:
        # Thinking block present but no closing tag — strip everything up to
        # the first JSON-like structure
        cleaned = cleaned.split("<unused94>", 1)[0].strip()

    # ── Strip markdown code fences ──
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed — attempting truncation recovery")
        cleaned = attempt_json_recovery(cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Recovery failed: {e}")
            return {
                "individual_signals": [],
                "combination_patterns": [],
                "overall_assessment": None,
                "urgency": "routine",
                "analysis_confidence": "low",
                "parse_error": str(e),
                "raw_response": raw_response
            }
    
    # ── Validate top-level keys ──
    defaults = {
        "individual_signals": [],
        "combination_patterns": [],
        "overall_assessment": None,
        "urgency": "routine",
        "analysis_confidence": "low"
    }
    for key, default in defaults.items():
        if key not in parsed:
            parsed[key] = default

    # ── Validate each individual signal entry ──
    for i, signal in enumerate(parsed.get("individual_signals", [])):
        if not isinstance(signal, dict):
            parsed["individual_signals"][i] = {"signal": str(signal), "reason": None}
            continue
        if "signal" not in signal:
            signal["signal"] = None
        if "reason" not in signal:
            signal["reason"] = None

    # ── Validate each combination pattern entry ──
    pattern_defaults = {
        "signals_involved": [],
        "pattern_description": None,
        "co-occurrence_context": None,
        "clinical_significance": "low",
        "is_progressive": "false",
        "suggested_investigation": None
    }
    for i, pattern in enumerate(parsed.get("combination_patterns", [])):
        if not isinstance(pattern, dict):
            parsed["combination_patterns"][i] = dict(pattern_defaults)
            continue
        for key, default in pattern_defaults.items():
            if key not in pattern:
                pattern[key] = default

    # ── Normalize enums ──
    if parsed["urgency"] not in ("routine", "soon", "urgent"):
        parsed["urgency"] = "routine"
    if parsed["analysis_confidence"] not in ("high", "medium", "low"):
        parsed["analysis_confidence"] = "low"

    return parsed
