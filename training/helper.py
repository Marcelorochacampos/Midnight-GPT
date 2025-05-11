import torch
import torch.nn.functional as F
import argparse

def generate(model, config, prompt_ids, max_tokens):
    output_ids = prompt_ids
    for _ in range(max_tokens):
        if output_ids.shape[1] >= config["context_size"]:
            break
        with torch.no_grad():
            logits = model(output_ids)

        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)

        next_token_id = torch.multinomial(probs, num_samples=1)

        output_ids = torch.cat([output_ids, next_token_id], dim=-1)

    return output_ids


def generate_with_prompt(model, config, tokenizer, prompt, max_tokens=100):
    model.eval()

    prompt = tokenizer.encode(prompt).unsqueeze(dim=0).to("cuda")

    return tokenizer.decode(generate(model, config, prompt, max_tokens=max_tokens).flatten())


def get_cli_args():
    parser = argparse.ArgumentParser(description="The model training arg helper.")

    parser.add_argument("--dataset", type=str, help="Full dataset path")
    parser.add_argument("--tokenizer", type=str, help="Full tokenizer path")
    parser.add_argument("--model", type=str, help="Full model path")
    parser.add_argument("--huggingface", type=str, help="Dataset name from huggingface")
    parser.add_argument("--subset", type=str, help="Subset/config name (e.g., emoji, emotion, sentiment, etc.)")

    return parser.parse_args()