import argparse
import json
import os
from pathlib import Path

from PIL import Image
import torch
from tqdm import tqdm


PROMPT = (
    "In under 15 words, describe this garment: color, pattern, sleeve length, fit."
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Caption cloth images with LLaVA-1.5-7B")
    p.add_argument("--cloth_dir",   required=True,  help="Path to folder of cloth images")
    p.add_argument("--output_json", required=True,  help="Path to write captions.json")
    p.add_argument("--resume",      action="store_true", help="Skip already-captioned images")
    return p.parse_args()


def available_vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


def load_model():
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    use_4bit = available_vram_gb() < 16.0

    print(f"  VRAM available: {available_vram_gb():.1f} GB  →  4-bit loading: {use_4bit}")

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return processor, model


def caption_image(path: Path, processor, model) -> str:
    image = Image.open(path).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def load_existing(output_json: str) -> dict:
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            return json.load(f)
    return {}


def save(captions: dict, output_json: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(captions, f, indent=2)


def main():
    args = parse_args()

    cloth_dir = Path(args.cloth_dir)
    if not cloth_dir.is_dir():
        raise SystemExit(f"ERROR: --cloth_dir does not exist: {cloth_dir}")

    image_paths = sorted(
        p for p in cloth_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise SystemExit(f"ERROR: no images found in {cloth_dir}")

    captions = load_existing(args.output_json)
    failed_log = Path(args.output_json).with_name("failed_captions.txt")

    if args.resume and captions:
        before = len(image_paths)
        image_paths = [p for p in image_paths if p.name not in captions]
        print(f"Resume: skipping {before - len(image_paths)} already-captioned images, "
              f"{len(image_paths)} remaining.")

    if not image_paths:
        print("Nothing to do — all images already captioned.")
        return

    print(f"Loading model…")
    processor, model = load_model()
    print(f"Model ready. Captioning {len(image_paths)} images.\n")

    failed = []
    save_every = 50

    for i, path in enumerate(tqdm(image_paths, desc="Captioning", unit="img")):
        try:
            cap = caption_image(path, processor, model)
            captions[path.name] = cap
        except Exception as e:
            tqdm.write(f"  FAILED {path.name}: {e}")
            failed.append(path.name)

        if (i + 1) % save_every == 0:
            save(captions, args.output_json)
            tqdm.write(f"  Checkpoint saved ({len(captions)} captions so far)")

    # Final save
    save(captions, args.output_json)
    print(f"\nDone. {len(captions)} captions written to {args.output_json}")

    if failed:
        with open(failed_log, "w") as f:
            f.write("\n".join(failed))
        print(f"{len(failed)} failed images logged to {failed_log}")


if __name__ == "__main__":
    main()
