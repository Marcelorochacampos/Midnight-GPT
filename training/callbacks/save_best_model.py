import torch
from transformers import TrainerCallback

class SaveBestModelCallback(TrainerCallback):
    def __init__(self, model_path):
        self.best_loss = float("inf")
        self.model_path = model_path

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None:
            return
        
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            print(f"\nNew best model! eval_loss: {eval_loss:.4f}")
            torch.save(model.state_dict(), self.model_path)