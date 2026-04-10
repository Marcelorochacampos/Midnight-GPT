import os
import torch

def save_checkpoint(model, optimizer, scaler, step, best_params, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "best_params": best_params
    }, path)

def load_checkpoint(model, optimizer, scaler):
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, None

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0), ckpt.get("best_params", None)