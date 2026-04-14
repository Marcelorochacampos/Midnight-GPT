# Midnight-GPT

Midnight-GPT is a case study on training a GPT-style decoder-only Transformer from scratch under real-world hardware constraints. The project combines a custom PyTorch model implementation, dataset preparation scripts, Optuna-based hyperparameter exploration, Hugging Face training pipelines, inference utilities, a FastAPI backend, and a small Next.js frontend.

The main goal of this repository is not to present a state-of-the-art model. It is to document the engineering process, tradeoffs, bottlenecks, and failure modes involved in building and training a small language model end to end on a single consumer GPU.

## Project Summary

- Model type: decoder-only Transformer implemented from scratch in PyTorch
- Tokenizer: GPT-2 tokenizer from Hugging Face
- Core configuration: 10 layers, 12 attention heads, 768 embedding dimension, 512 token context window
- Approximate size: ~148M parameters in the current implementation
- Training infrastructure: Hugging Face Datasets + Trainer, MLflow, Optuna, custom callbacks
- Hardware used: NVIDIA RTX 3060 Ti
- Total training budget: roughly 2 billion tokens across 4 training iterations
- Total runtime: roughly 90 hours
- Final outcome: useful as a learning and engineering case study, but not strong enough to be considered a production-quality language model

## Why This Project Exists

This repository became a practical study of what happens when you try to train a GPT-like model from zero with limited compute, limited storage, and limited iteration speed. That makes it valuable for a different reason: it captures the gap between a clean theoretical training pipeline and the messy reality of data quality issues, checkpoint constraints, optimizer tuning, and catastrophic forgetting across multiple fine-tuning stages.

## Repository Layout

```text
.
├── api/                 # FastAPI inference service
├── front/               # Next.js frontend
├── training/
│   ├── config/          # Global configuration
│   ├── data-pipeline/   # Dataset download and tokenization scripts
│   ├── dataset/         # Local tokenized and untokenized datasets
│   ├── checkpoints/     # Optuna results and saved training checkpoints
│   ├── mlruns/          # MLflow experiment tracking artifacts
│   ├── model/           # Transformer architecture implementation
│   ├── training_pipeline/
│   │   ├── callbacks/   # Token counting, text generation, best-model saving
│   │   ├── collate/
│   │   ├── dataset/
│   │   └── trainer/
│   ├── optuna.py        # Hyperparameter search
│   ├── inference.py     # Local CLI inference
│   ├── training[...].py # Training iterations for different corpora
│   └── utils/
└── README.md
```

## Model Architecture

Midnight-GPT uses a custom decoder-only Transformer stack written directly in PyTorch. The architecture includes token embeddings, learned positional embeddings, stacked Transformer blocks, causal self-attention, feed-forward layers, layer normalization, and a final projection head over the vocabulary.

The current default configuration in the repository is:

```yaml
vocabulary_size: 50257
context_size: 512
embedding_dim: 768
heads_num: 12
layers_num: 10
dropout_rate: 0.1
use_bias: false
```

This keeps the project large enough to surface real training problems, but still small enough to be trainable on a single RTX 3060 Ti with aggressive tradeoffs.

## Training Case Study

### 1. Data Collection and Preprocessing

The data collection process started by selecting datasets from the Hugging Face Hub that seemed likely to improve the final model. Instead of building one monolithic pipeline, the project uses separate scripts to download and tokenize each dataset independently. That made experimentation easier and reduced coupling between data sources.

One major design decision was to keep tokenized datasets already serialized on disk instead of tokenizing on the fly during training. The reasoning was straightforward: avoid dataset streaming latency and avoid tokenization overhead during each training step so the GPU would spend less time idle. On constrained hardware, this kind of systems decision matters.

That choice solved one bottleneck but exposed another one: storage pressure. Large corpora became harder to manage because pre-tokenized datasets occupy a significant amount of disk space, especially when split into train, validation, and test partitions.

On the data quality side, preprocessing stayed intentionally light. There was no serious feature engineering or advanced dataset curation. The cleaning work was limited mostly to small regex-based cleanup steps, such as removing very noisy HTML fragments. In hindsight, this was one of the main weaknesses of the whole training effort.

The model was still exposed to many low-value records, including:

- very short texts
- noisy web fragments
- weakly informative tables
- other low-signal samples that diluted the useful training signal

This became visible later in training quality: the model could sometimes produce locally coherent continuations, but struggled to maintain quality across the full output.

### 2. Hyperparameter Exploration with Optuna

Before the first full training iteration, Optuna was used to search for a more promising initial hyperparameter combination. Because the available hardware budget was small, the search space also had to stay small. Only a narrow parameter window and a limited number of trials were feasible.

The search focused on practical high-impact variables such as:

- learning rate
- batch size
- weight decay
- gradient accumulation

This was not an exhaustive search and was never meant to be. The goal was simply to avoid walking blindly into a very poor first configuration.

That is an important distinction: on limited hardware, hyperparameter optimization is often about reducing avoidable mistakes, not about finding a globally optimal setup.

### 3. Training Iterations

After establishing an initial set of hyperparameters, the training pipeline was expanded into four separate iterations, each using a different corpus strategy:

| Iteration | Dataset strategy | Purpose |
| --- | --- | --- |
| 1 | Wikipedia | Establish a cleaner and more stable baseline |
| 2 | Common Crawl | Increase coverage and linguistic variety |
| 3 | brWaC | Add a Portuguese-oriented web corpus |
| 4 | Wikipedia + FinePDF mix | Test whether a mixed dataset could improve signal quality |

Each new training phase required additional tuning decisions, especially around learning rate, in order to keep loss moving in the right direction without damaging what had already been learned. In practice, this meant making iterative adjustments to reduce the risk of catastrophic forgetting while still allowing the model to adapt to new corpora.

This part of the project reflects a very common reality in small-scale LLM work: once you leave the first training run, there is no single static hyperparameter recipe that continues to work unchanged.

### 4. Training Stack and Monitoring

The training runs relied heavily on Hugging Face tooling, which made it possible to use several useful techniques out of the box without rebuilding everything manually. The main ones used in this project were:

- gradient accumulation
- gradient clipping
- cosine learning rate scheduling with warmup
- mixed precision when CUDA was available
- evaluation on a step schedule

On top of that, the project includes custom callbacks for operational visibility during training. These callbacks were used to:

- count processed tokens during training
- generate sample text during evaluation steps
- save the best model according to the lowest `eval_loss`

Checkpoint storage was another real constraint. Keeping many checkpoints was not practical because of file size, so the project favored a more storage-aware approach: always preserve the best model seen so far instead of accumulating a large checkpoint history.

MLflow was also used to track experiments and metrics during the training process.

## Outcome

Across the four iterations, Midnight-GPT was trained on roughly 2 billion tokens over about 90 hours on a single RTX 3060 Ti.

The final result was mixed.

The model did show signs of learning. In many generations, it could produce short spans with reasonable local semantic consistency. At the sentence or paragraph fragment level, parts of the output sometimes looked promising.

However, the model did not reach an acceptable overall quality bar. Long-form generations frequently lost coherence, drifted semantically, or collapsed into low-information continuations. Full outputs only rarely held together in a convincing way from beginning to end.

Given the hardware limitations, storage constraints, and available training time, the project was concluded as a case study rather than extended indefinitely in search of marginal improvements.

## Main Lessons Learned

- Data quality mattered more than expected. Light cleanup was not enough for noisy web-scale sources.
- Pre-tokenizing datasets to disk helped keep the GPU busy, but it created a storage-management problem.
- Even modest hyperparameter search was valuable, but limited compute sharply restricted how much exploration was possible.
- Sequential training across different corpora required careful learning-rate adaptation to reduce catastrophic forgetting.
- Operational tooling such as token counters, text-generation callbacks, and best-checkpoint saving was essential for monitoring progress on long runs.
- Hardware constraints were not just an inconvenience; they shaped nearly every technical decision in the project.

## Running the Project

### Python environment

Create a virtual environment and install the main libraries used by the training and inference code:

```bash
pip install torch transformers datasets optuna mlflow fastapi uvicorn pyyaml
```

Depending on your CUDA setup, you may need a PyTorch installation command specific to your GPU and driver version.

### Run local inference

From the `training/` directory:

```bash
python inference.py --prompt "O período medieval foi marcado pelo"
```

### Run the FastAPI inference service

From the project root:

```bash
uvicorn api.main:app --reload
```

### Run the frontend

From the `front/` directory:

```bash
npm install
npm run dev
```

### Training scripts

From the `training/` directory, the repository includes separate scripts for different experiments, including:

```bash
python optuna.py
python .\training[wikipedia].py
python .\training[cc].py
python .\training[brwac].py
python .\training[mix].py
```

## What This Repository Is Good For

This repository is useful if you want to study:

- how to structure a small GPT training project from scratch
- how training decisions change under consumer-GPU constraints
- how dataset quality can dominate outcomes even when the training stack is technically sound
- how to combine custom model code with Hugging Face tooling for faster iteration

## References

The project was strongly influenced by the following books:

- Build a Large Language Model (From Scratch)
- Natural Language Processing with Transformers

## Final Note

Midnight-GPT did not end as a high-quality language model. It did, however, succeed as an honest engineering case study. That is still a useful outcome. The repository documents what was built, what worked, what failed, and where the real bottlenecks appeared when training a Transformer language model from scratch with limited resources.
