import torch
import torch.nn.functional as F
import argparse

def generate(model, config, prompt_ids, tokenizer, max_tokens):
    output_ids = prompt_ids
    context_window = prompt_ids
    for _ in range(max_tokens):
        if context_window.shape[1] >= config["context_size"]:
            context_window = context_window[:, 1:]
            # break
        with torch.no_grad():
            logits = model(context_window)

        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)

        next_token_id = torch.multinomial(probs, num_samples=1)

        output_ids = torch.cat([output_ids, next_token_id], dim=-1)
        context_window = torch.cat([context_window, next_token_id], dim=-1)
        print(tokenizer.decode(output_ids = torch.cat([output_ids, next_token_id], dim=-1).flatten()))
    return output_ids


def generate_with_prompt(model, config, tokenizer, prompt, max_tokens=100):
    model.eval()

    prompt = tokenizer.encode(prompt).unsqueeze(dim=0).to("cuda")

    return tokenizer.decode(generate(model, config, prompt, tokenizer, max_tokens=max_tokens).flatten())


def get_cli_args():
    parser = argparse.ArgumentParser(description="The model training arg helper.")

    parser.add_argument("--dataset", type=str, help="Full dataset path")
    parser.add_argument("--tokenizer", type=str, help="Full tokenizer path")
    parser.add_argument("--model", type=str, help="Full model path")
    parser.add_argument("--huggingface", type=str, help="Dataset name from huggingface")
    parser.add_argument("--subset", type=str, help="Subset/config name (e.g., emoji, emotion, sentiment, etc.)")

    return parser.parse_args()