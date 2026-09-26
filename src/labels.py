"""Label set + per-label decision thresholds for the PII/PHI GLiNER pipeline.

GLiNER is zero-shot: a "label" is just a natural-language prompt string, so
"standard" and "custom" labels are handled identically by the model. This
file is the single source of truth shared by the eval notebook and the ONNX
export script, so the label order (which becomes the ONNX output class
order) never drifts between training/eval and deployment.
"""

# Standard PII/PHI label prompts.
STANDARD_LABELS: dict[str, float] = {
    "person": 0.5,
    "email address": 0.4,
    "phone number": 0.4,
    "social security number": 0.3,
    "date of birth": 0.4,
    "home address": 0.5,
    "credit card number": 0.3,
    "medical record number": 0.3,
    "diagnosis": 0.4,
    "medication": 0.4,
    "health insurance id number": 0.35,
}

# Add your own zero-shot prompts here. A descriptive phrase works as well as
# a short tag -- GLiNER was trained on natural-language entity descriptions.
CUSTOM_LABELS: dict[str, float] = {
    "patient identifier assigned by a hospital": 0.35,
    "clinical trial enrollment number": 0.35,
}

LABEL_THRESHOLDS: dict[str, float] = {**STANDARD_LABELS, **CUSTOM_LABELS}
ALL_LABELS: list[str] = list(LABEL_THRESHOLDS)
