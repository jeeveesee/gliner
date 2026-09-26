"""Export the GLiNER large PII/PHI model to ONNX for the Scala pipeline.

The full flattened prompt list from configs/taxonomy.json is baked into the
export only to produce trace-time example shapes -- the graph itself is
dynamic over sequence length, so the Scala side reconstructs the same label
prompt at inference time. taxonomy.json, thresholds.json, and the frozen
`prompts.json` (export class order) are copied into the export dir so
Scala never re-derives or hardcodes them.
"""

import json
import shutil
from pathlib import Path

from gliner import GLiNER

from taxonomy import load_taxonomy

MODEL_ID = "urchade/gliner_large-v2.1"
TAXONOMY_PATH = Path("../configs/taxonomy.json")
THRESHOLDS_PATH = Path("../configs/thresholds.json")
SAVE_DIR = Path("onnx_export")


def main() -> None:
    taxonomy = load_taxonomy(TAXONOMY_PATH, THRESHOLDS_PATH)

    model = GLiNER.from_pretrained(MODEL_ID)
    model.eval()

    result = model.export_to_onnx(
        save_dir=SAVE_DIR,
        onnx_filename="gliner_pii.onnx",
        quantize=True,  # also writes gliner_pii_quantized.onnx (int8, CPU-friendly)
        opset=19,
        labels=taxonomy.all_prompts,
        text="Patient data placeholder used only to trace the export graph.",
    )
    print("Exported:", result)

    (SAVE_DIR / "prompts.json").write_text(json.dumps(taxonomy.all_prompts, indent=2))
    shutil.copy(TAXONOMY_PATH, SAVE_DIR / "taxonomy.json")
    shutil.copy(THRESHOLDS_PATH, SAVE_DIR / "thresholds.json")
    # gliner_config.json (ent_token, sep_token, max_width, ...) and the
    # tokenizer files are already written into SAVE_DIR by export_to_onnx.


if __name__ == "__main__":
    main()
