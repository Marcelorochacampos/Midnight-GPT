import os
import gc
import torch
import optuna
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from model.architecture import MidnightGPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 500
EVAL_INTERVAL = 100
FINAL_TRAIN_STEPS = 150000

CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "last.pt")

GEN_PROMPT = "Durante o período medieval "

def create_loader(dataset, batch_size, shuffle, num_workers=2, persistent=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=persistent
    )

def create_model(CONFIG):
    return MidnightGPT(CONFIG).to(device)

def train_step(model, batch, optimizer, scaler):
    model.train()
    input_ids = batch["input_ids"].to(device)

    with autocast(device_type="cuda"):
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            input_ids[:, 1:].contiguous().view(-1)
        )

    scaler.scale(loss).backward()
    return loss

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses = []

    for i, batch in enumerate(loader):
        if i > 20:
            break

        input_ids = batch["input_ids"].to(device)

        with autocast(device_type="cuda"):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                input_ids[:, 1:].contiguous().view(-1)
            )

        losses.append(loss.item())

    return sum(losses) / len(losses)

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    CONFIG = {
        "vocabulary_size": 50257,
        "context_size": 512,
        "embedding_dim": 768,
        "heads_num": 12,
        "layers_num": 10,
        "dropout_rate": 0.1,
        "use_bias": False
    }

    CONFIG["head_size"] = CONFIG["embedding_dim"] // CONFIG["heads_num"]

    params = (
        CONFIG["vocabulary_size"] * CONFIG["embedding_dim"] +
        CONFIG["context_size"] * CONFIG["embedding_dim"] +
        CONFIG["layers_num"] * (12 * CONFIG["embedding_dim"] * CONFIG["embedding_dim"])
    )

    print("==============================")
    print("Model configuration:", CONFIG)
    print(f"Model size (parameters): {params:,}")
    print("==============================")

    train_dataset = load_from_disk("./dataset/tokenized/train_512")
    val_dataset = load_from_disk("./dataset/tokenized/val_512")
    test_dataset = load_from_disk("./dataset/tokenized/test_512")

    train_dataset.set_format(type="torch", columns=["input_ids"])
    val_dataset.set_format(type="torch", columns=["input_ids"])
    test_dataset.set_format(type="torch", columns=["input_ids"])

    def objective(trial):
        try:
            print("\n", torch.cuda.memory_allocated() / 1e9, "GB allocated [Before trial]")
            print(torch.cuda.memory_reserved() / 1e9, "GB reserved [Before trial]")
            lr = trial.suggest_float("lr", 2e-5, 2e-4, log=True)
            batch_size = trial.suggest_categorical("batch_size", [4, 8])
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            grad_accum = trial.suggest_categorical("grad_accum", [2, 4])

            train_loader = create_loader(train_dataset, batch_size, True, persistent=False)
            val_loader = create_loader(val_dataset, batch_size, False, persistent=False)

            model = create_model(CONFIG)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scaler = GradScaler("cuda")

            optimizer.zero_grad(set_to_none=True)

            step = 0

            for batch in train_loader:
                loss = train_step(model, batch, optimizer, scaler)

                if (step + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                if step % EVAL_INTERVAL == 0 and step > 0:
                    val_loss = evaluate(model, val_loader)
                    trial.report(val_loss, step)

                    print(f"[Trial {trial.number}] Step {step} | Val Loss: {val_loss:.4f}")

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                if step % 20 == 0:
                    print(f"[Trial {trial.number}] Step {step} | Train Loss: {loss.item():.4f}")

                step += 1
                if step >= MAX_STEPS:
                    break

            return evaluate(model, val_loader)

        finally:
            print("\n",torch.cuda.memory_allocated() / 1e9, "GB allocated [Before cleanup - After trial]")
            print(torch.cuda.memory_reserved() / 1e9, "GB reserved [Before cleanup - After trial]")
            del model, optimizer, scaler, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
            print("\n",torch.cuda.memory_allocated() / 1e9, "GB allocated [After cleanup]")
            print(torch.cuda.memory_reserved() / 1e9, "GB reserved [After cleanup]")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,
            n_warmup_steps=2,
            interval_steps=1
        ),
        sampler=optuna.samplers.TPESampler(n_startup_trials=3)
    )

    study.optimize(objective, n_trials=10, show_progress_bar=True)

    best_params = study.best_trial.params
    print("Best params:", best_params)
    #save



if __name__ == "__main__":
    main()