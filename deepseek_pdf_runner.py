import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

try:
    import pypdfium2 as pdfium
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "pypdfium2 is required for PDF rasterization. Install it via `pip install pypdfium2`."
    ) from exc


DTYPE_MAP: Dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch DeepSeek-OCR inference over PDF pages.")
    parser.add_argument("--pdf", type=Path, required=True, help="Input PDF file")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for Markdown + images")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR", help="Model repo or local path")
    parser.add_argument(
        "--prompt",
        default="<image>\n<|grounding|>Convert the document to markdown.",
        help="Prompt sent before each page",
    )
    parser.add_argument("--base-size", type=int, default=1024, help="Model base_size argument")
    parser.add_argument("--image-size", type=int, default=640, help="Model image_size argument")
    parser.add_argument("--no-crop-mode", action="store_true", help="Disable crop_mode flag")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep rendered page images")
    parser.add_argument("--test-compress", action="store_true", help="Enable test_compress flag")
    parser.add_argument("--save-results", action="store_true", help="Pass save_results=True to infer()")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda, cuda:0, cpu)")
    parser.add_argument(
        "--precision",
        choices=list(DTYPE_MAP.keys()),
        default="bfloat16",
        help="Computation dtype",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Rasterization DPI for PDF pages")
    parser.add_argument(
        "--image-format",
        choices=["png", "jpg"],
        default="png",
        help="Image format for temporary renders",
    )
    parser.add_argument("--concat", action="store_true", help="Write combined_markdown.md with all pages")
    return parser.parse_args()


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int, image_format: str) -> List[Path]:
    """Convert each PDF page to an image file compatible with DeepSeek's infer()."""
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    image_paths: List[Path] = []
    for index in range(len(doc)):
        page = doc[index]
        pil_image = page.render(scale=scale).to_pil()
        page_path = out_dir / f"page_{index+1:04d}.{image_format}"
        pil_image.save(page_path, format=image_format.upper())
        image_paths.append(page_path)
    return image_paths


def extract_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "preds", "result", "markdown"):
            if key in result and isinstance(result[key], str):
                return result[key]
        return json.dumps(result, ensure_ascii=False)
    if isinstance(result, (list, tuple)):
        return "\n".join(str(item) for item in result)
    return str(result)


def load_tokenizer_and_model(
    model_name: str,
    device: str,
    precision: str,
) -> Tuple[AutoTokenizer, AutoModel]:
    torch_dtype = DTYPE_MAP[precision]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation="flash_attention_2",
    )
    model = model.eval().to(device=device, dtype=torch_dtype)
    return tokenizer, model


def run_pdf_to_markdown(
    *,
    pdf_path: Path,
    out_dir: Path,
    model_name: str = "deepseek-ai/DeepSeek-OCR",
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    keep_intermediate: bool = False,
    test_compress: bool = False,
    save_results: bool = False,
    device: str = "cuda",
    precision: str = "bfloat16",
    dpi: int = 300,
    image_format: str = "png",
    concat: bool = False,
    write_files: bool = True,
    tokenizer: Optional[AutoTokenizer] = None,
    model: Optional[AutoModel] = None,
) -> Dict[str, Any]:
    """Run DeepSeek-OCR over every PDF page, optionally writing Markdown files."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "pages"
    image_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir = out_dir / "markdown"
    if write_files:
        markdown_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DeepSeek-OCR] Rasterizing {pdf_path} ...")
    page_images = pdf_to_images(pdf_path, image_dir, dpi, image_format)
    print(f"[DeepSeek-OCR] Converted {len(page_images)} pages -> images")

    if tokenizer is None or model is None:
        tokenizer, model = load_tokenizer_and_model(model_name, device, precision)

    combined_markdown: List[str] = []
    per_page_markdown: List[str] = []

    for page_id, image_path in enumerate(page_images, start=1):
        print(f"[DeepSeek-OCR] Page {page_id}/{len(page_images)} -> {image_path.name}")
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=str(out_dir),
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            test_compress=test_compress,
            save_results=save_results,
        )
        markdown_text = extract_text(result)
        if not markdown_text:
            print(f"[DeepSeek-OCR] Warning: empty output for page {page_id}")
        if write_files:
            page_md_path = markdown_dir / f"page_{page_id:04d}.md"
            page_md_path.write_text(markdown_text, encoding="utf-8")
        combined_markdown.append(f"<!-- Page {page_id} -->\n{markdown_text}\n")
        per_page_markdown.append(markdown_text)

    combined_text = "\n".join(combined_markdown) if combined_markdown else ""
    if concat and write_files:
        concat_path = out_dir / "combined_markdown.md"
        concat_path.write_text(combined_text, encoding="utf-8")
        print(f"[DeepSeek-OCR] Wrote combined markdown to {concat_path}")

    if not keep_intermediate:
        for image_path in page_images:
            try:
                image_path.unlink()
            except FileNotFoundError:
                pass
        if not any(image_dir.iterdir()):
            image_dir.rmdir()

    print("[DeepSeek-OCR] Done.")

    return {
        "num_pages": len(page_images),
        "page_markdown": per_page_markdown,
        "combined_markdown": combined_text if concat else "",
        "output_dir": str(out_dir),
        "markdown_dir": str(markdown_dir) if write_files else "",
    }


def main():
    args = parse_args()
    run_pdf_to_markdown(
        pdf_path=args.pdf,
        out_dir=args.out_dir,
        model_name=args.model,
        prompt=args.prompt,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=not args.no_crop_mode,
        keep_intermediate=args.keep_intermediate,
        test_compress=args.test_compress,
        save_results=args.save_results,
        device=args.device,
        precision=args.precision,
        dpi=args.dpi,
        image_format=args.image_format,
        concat=args.concat,
        write_files=True,
    )


if __name__ == "__main__":
    main()
