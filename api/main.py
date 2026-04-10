from pathlib import Path
import sys

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_ROOT = PROJECT_ROOT / "training"
CONFIG_PATH = TRAINING_ROOT / "config" / "global_configuration.yaml"

# Make training modules importable from api/main.py
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from model.architecture import MidnightGPT
from utils.configuration import load_global_configuration


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int | None = Field(default=None, ge=1, le=1024)
    temperature: float | None = Field(default=None, gt=0.0, le=5.0)


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    max_new_tokens: int
    temperature: float
    model_path: str
    device: str


app = FastAPI(title="Midnight-GPT Inference API", version="1.0.0")

model: MidnightGPT | None = None
tokenizer = None
model_config: dict | None = None
inference_config: dict | None = None
resolved_model_path: Path | None = None
device: torch.device | None = None


def _resolve_model_path(raw_model_path: str) -> Path:
    path = Path(raw_model_path)
    if path.is_absolute():
        return path
    return TRAINING_ROOT / path


def _generate_text(prompt: str, max_new_tokens: int, temperature: float) -> str:
    if model is None or tokenizer is None or model_config is None or device is None:
        raise RuntimeError("Model is not initialized")

    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_window = input_ids[:, -model_config["context_size"] :]
            logits = model(input_ids=input_window).logits
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


@app.on_event("startup")
def startup_event() -> None:
    global model, tokenizer, model_config, inference_config, resolved_model_path, device

    config = load_global_configuration(CONFIG_PATH)
    model_config = config["model_architecture"]
    inference_config = config.get("inference", {})
    resolved_model_path = _resolve_model_path(config["paths"]["model"]["dev_weights"])

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {resolved_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    model = MidnightGPT(model_config).to(device)
    state_dict = torch.load(resolved_model_path, map_location=device)
    model.load_state_dict(state_dict)


@app.get("/health")
def health() -> dict:
    is_ready = model is not None and tokenizer is not None
    return {
        "status": "ok" if is_ready else "loading",
        "ready": is_ready,
        "device": str(device) if device is not None else None,
        "model_path": str(resolved_model_path) if resolved_model_path is not None else None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    if model is None or tokenizer is None or inference_config is None or resolved_model_path is None or device is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    try:
        max_new_tokens = request.max_new_tokens or inference_config.get("max_new_tokens", 100)
        temperature = request.temperature or inference_config.get("temperature", 0.8)

        generated = _generate_text(
            prompt=request.prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            model_path=str(resolved_model_path),
            device=str(device),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
