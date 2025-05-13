import torch
import torch.nn.functional as F
from datetime import datetime
from torch.cuda.amp import autocast
from torch.amp.grad_scaler import GradScaler
from helper import generate_with_prompt

EPOCHS = 1

def train(model, optimizer, tokenizer, dataloader, config, training_config):
    scaler = GradScaler(device="cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(training_config.get("epochs", EPOCHS)):
        torch.cuda.empty_cache()

        for step_num, sample in enumerate(dataloader):
            model.train()
            input, targets = sample
            input, targets = input.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = model(input)
                logits_view = logits.reshape(-1, config["vocabulary_size"])
                targets_view = targets.reshape(-1)
                loss = F.cross_entropy(logits_view, targets_view)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"Step {step_num}. Loss {loss.item():.3f}")

            if step_num > 0 and step_num % training_config["evaluation_interval"] == 0:
                print("Model GPT:\n" + generate_with_prompt(model, config, tokenizer, "\n"))
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"model_mad_{timestamp}_{step_num}.pth"
                path = f"./checkpoints/model/{model_name}"
                print(f"Storing checkpoint: {path}\n")
                torch.save(model.state_dict(), path)

        print(f"Epoch: {epoch}")
        print("Model GPT:\n" + generate_with_prompt(model, config, tokenizer, "\n"))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_mad_{timestamp}_{epoch}.pth"
        path = f"./checkpoints/model/{model_name}"
        print(f"Storing checkpoint: {path}\n")
        torch.save(model.state_dict(), path)