# DeepSeek-OCR GPU Setup (A4000)

This repo contains helper instructions and a Python script to run [`deepseek-ai/DeepSeek-OCR`](https://huggingface.co/deepseek-ai/DeepSeek-OCR) with an NVIDIA RTX A4000. It follows the tested versions from the Hugging Face card and adds a PDF-to-Markdown workflow.

## 1. Environment

```bash
# Optional: manage with uv (recommended)
uv venv .venv
source .venv/bin/activate

# Core deps from HF card
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.46.3 tokenizers==0.20.3 einops addict easydict
pip install flash-attn==2.7.3 --no-build-isolation
# PDF helpers
pip install pypdfium2 Pillow
# Optional REST API deps
pip install fastapi uvicorn python-multipart
```

If you plan to use vLLM, install its nightly wheels (needs CUDA 12.1 driver runtime):

```bash
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

## 2. GPU prerequisites

- Driver that supports CUDA 11.8+ (A4000 works with 535+ drivers).
- Set `CUDA_VISIBLE_DEVICES` if you have multiple GPUs.
- Mixed-precision works best in `bfloat16`; older GPUs can switch to `torch.float16` in the script.

## 3. PDF workflow

`deepseek_pdf_runner.py` converts each PDF page to an image, feeds it to the model with the markdown conversion prompt, and writes one Markdown file per page plus a concatenated file.

Run it like this:

```bash
python deepseek_pdf_runner.py \
  --pdf /path/to/file.pdf \
  --out-dir outputs/my_pdf \
  --device cuda \
  --precision bfloat16 \
  --base-size 1024 \
  --image-size 640 \
  --test-compress --save-results --concat
```

See the script for more knobs (prompt text, compression toggle, keeping intermediate images, etc.).

## 4. RTX A4000 tuning tips

- Export `CUDA_VISIBLE_DEVICES=0` (or your preferred index) before running if the machine hosts multiple GPUs.
- The A4000 (16 GB VRAM) comfortably handles `base_size=1024` / `image_size=640`. Drop to `base_size=640` for extremely long documents or when batching with vLLM.
- Use `--precision bfloat16` for best quality; switch to `float16` only if the driver/toolchain combination complains.
- `--test-compress` reduces VRAM spikes at the cost of a small speed penalty; keep it on when you see CUDA OOMs.
- Add `--keep-intermediate` to inspect the per-page renders saved under `out-dir/pages`.

## 5. Simple REST API (upload a PDF)

- Ensure the environment is activated and dependencies above (plus `fastapi uvicorn python-multipart`) are installed.
- Start the server: `uvicorn api_server:app --host 0.0.0.0 --port 8000`.
- Health probe: `curl http://localhost:8000/health`.
- OCR request:

```bash
curl -X POST "http://localhost:8000/ocr/pdf" \
  -F "file=@/path/to/file.pdf" \
  -F "prompt=<image>\\n<|grounding|>Convert the document to markdown." \
  -F "base_size=1024" -F "image_size=640"
```

Response contains per-page markdown plus `combined_markdown`. Model weights are loaded once per process and cached in GPU memory for subsequent uploads.
