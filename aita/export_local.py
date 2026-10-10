"""Download Aita's LoRA weights from Tinker so she runs on your own machine.

Usage:
    python -m aita.export_local              # PEFT adapter only (megabytes)
    python -m aita.export_local --merge      # also a full merged Hugging Face model (gigabytes)

After --merge, convert to GGUF with llama.cpp and load it in Ollama (see README):
    python llama.cpp/convert_hf_to_gguf.py models/merged --outfile models/aita.gguf --outtype q8_0
    ollama create aita -f models/Modelfile
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from aita.train_tinker import DEFAULT_RUN_FILE, ROOT

DEFAULT_MODELS_DIR = ROOT / "models"
MODELFILE = """FROM ./aita.gguf
PARAMETER temperature 0.4
PARAMETER num_ctx 4096
"""


def export(args: argparse.Namespace) -> None:
    from tinker_cookbook import weights

    run = json.loads(args.run_file.read_text(encoding="utf-8"))
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {run['tinker_path']} ...")
    raw_dir = weights.download(tinker_path=run["tinker_path"], output_dir=str(out / "adapter_raw"))

    peft_dir = out / "peft_adapter"
    weights.build_lora_adapter(base_model=run["base_model"], adapter_path=raw_dir, output_path=str(peft_dir))
    print(f"PEFT adapter: {peft_dir}")

    if args.merge:
        merged_dir = out / "merged"
        weights.build_hf_model(base_model=run["base_model"], adapter_path=raw_dir, output_path=str(merged_dir))
        (out / "Modelfile").write_text(MODELFILE, encoding="utf-8")
        print(f"Merged model: {merged_dir}\nNext: convert to GGUF, then `ollama create aita -f {out / 'Modelfile'}`")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-file", type=Path, default=DEFAULT_RUN_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--merge", action="store_true", help="also build a merged full model")
    return parser.parse_args(argv)


if __name__ == "__main__":
    export(parse_args())
