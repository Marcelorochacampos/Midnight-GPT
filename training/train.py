import torch
import torch.nn.functional as F
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast

from helper import generate_with_prompt

EPOCHS = 5

def train(model, optimizer, tokenizer, dataloader, config, training_config):
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()

        for step_num, sample in enumerate(dataloader):
            model.train()
            input, targets = sample
            input, targets = input.to("cuda"), targets.to("cuda")

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = model(input)

                # logits_view = logits.view(training_config["batch_size"] * config["context_size"], config["vocabulary_size"])
                # targets_view = targets.view(training_config["batch_size"] * config["context_size"])
                logits_view = logits.reshape(-1, config["vocabulary_size"])
                targets_view = targets.reshape(-1)

                loss = F.cross_entropy(logits_view, targets_view)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"Step {step_num}. Loss {loss.item():.3f}")

        print("Epoch: {}".format(epoch))
        print("Model GPT:\n" + generate_with_prompt(model, config, tokenizer, "\n"))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "model_mad_{}_{}.pth".format(timestamp, epoch)
        path = "./checkpoints/model/{}".format(model_name)
        print("Storing checkpoint: {}\n".format(path))
        torch.save(model.state_dict(), path)


