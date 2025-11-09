import tempfile
from pathlib import Path
from typing import Dict, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse

from deepseek_pdf_runner import load_tokenizer_and_model, run_pdf_to_markdown

app = FastAPI(title="DeepSeek OCR API", version="0.1.0")

MODEL_CACHE: Dict[Tuple[str, str, str], Tuple[object, object]] = {}


def get_cached_model(model_name: str, device: str, precision: str):
    cache_key = (model_name, device, precision)
    if cache_key not in MODEL_CACHE:
        tokenizer, model = load_tokenizer_and_model(model_name, device, precision)
        MODEL_CACHE[cache_key] = (tokenizer, model)
    return MODEL_CACHE[cache_key]


@app.post("/ocr/pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    model_name: str = Form("deepseek-ai/DeepSeek-OCR"),
    prompt: str = Form("<image>\n<|grounding|>Convert the document to markdown."),
    base_size: int = Form(1024),
    image_size: int = Form(640),
    crop_mode: bool = Form(True),
    test_compress: bool = Form(True),
    save_results: bool = Form(True),
    concat: bool = Form(True),
    precision: str = Form("bfloat16"),
    device: str = Form("cuda"),
    dpi: int = Form(300),
    image_format: str = Form("png"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    if precision not in {"bfloat16", "float16", "float32"}:
        raise HTTPException(status_code=400, detail="Unsupported precision value.")

    tokenizer, model = get_cached_model(model_name, device, precision)

    pdf_bytes = await file.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / file.filename
        pdf_path.write_bytes(pdf_bytes)
        out_dir = Path(tmpdir) / "outputs"

        run_result = run_pdf_to_markdown(
            pdf_path=pdf_path,
            out_dir=out_dir,
            model_name=model_name,
            prompt=prompt,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            keep_intermediate=False,
            test_compress=test_compress,
            save_results=save_results,
            device=device,
            precision=precision,
            dpi=dpi,
            image_format=image_format,
            concat=concat,
            write_files=False,
            tokenizer=tokenizer,
            model=model,
        )

    return JSONResponse(
        {
            "num_pages": run_result["num_pages"],
            "pages": run_result["page_markdown"],
            "combined_markdown": run_result["combined_markdown"],
        }
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}
