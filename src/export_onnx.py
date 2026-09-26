"""Export the GLiNER large PII/PHI label set to ONNX for the Scala pipeline.

Uses gliner's built-in `export_to_onnx` (gliner>=0.2.x). The label list is
baked into the export only to produce trace-time example shapes -- the
resulting graph is fully dynamic over sequence length, so the Scala side
must reconstruct the *same* label prompt (same labels, same order) around
each real input text at inference time. `labels.json` and `thresholds.json`
are written next to the model specifically so Scala never hardcodes them.
"""

import json
from pathlib import Path

from gliner import GLiNER

from labels import ALL_LABELS, LABEL_THRESHOLDS

MODEL_ID = "urchade/gliner_large-v2.1"
SAVE_DIR = Path("onnx_export")


def main() -> None:
    model = GLiNER.from_pretrained(MODEL_ID)
    model.eval()

    result = model.export_to_onnx(
        save_dir=SAVE_DIR,
        onnx_filename="gliner_pii.onnx",
        quantize=True,  # also writes gliner_pii_quantized.onnx (int8, CPU-friendly)
        opset=19,
        labels=ALL_LABELS,
        text="Patient data placeholder used only to trace the export graph.",
    )
    print("Exported:", result)

    (SAVE_DIR / "labels.json").write_text(json.dumps(ALL_LABELS, indent=2))
    (SAVE_DIR / "thresholds.json").write_text(json.dumps(LABEL_THRESHOLDS, indent=2))
    # gliner_config.json (ent_token, sep_token, max_width, ...) and the
    # tokenizer files are already written into SAVE_DIR by export_to_onnx.


if __name__ == "__main__":
    main()
