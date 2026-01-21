## ⚡ Installation & Setup

### 1️⃣ Install `uv`

If you don’t have `uv` installed, install it using:

#### Recommended:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Or you can try:

```bash
pip install uv
```

#### Clone the github repo and navigate to the project directory

### 2️⃣ install dependencies

```bash
uv sync
```

### 3️⃣ Run the main file

```bash
uv run main.py
```

#### Model download notice:

- Model download may fail due to cookies issues and not official drive download scripts.
- If download fails manually download model from drive

#### Drive link:

```
https://drive.google.com/file/d/1CSsTIRGM_REGw8V2Ey242VU3Pmk61Rqe/view?usp=sharing
```

- Download the model in the same directory as the main.py file
- Save as "sarcasm_finetuned_v2.pth"
- Then run the main.py file with:

```
uv run main.py
```

#### Note:

- This is a very small scale LLM so there may be grammatical errors
- Try changing the temperature value in the main.py file to get optimal output

## Model Architecture

### The main blocks of the model are:

- Token + Positional Embeddings

- Stacked Transformer Decoder Blocks

- Causal (masked) self-attention

- Feed-forward networks with GELU

- Pre-LayerNorm architecture

- Autoregressive text generation

### Architecture Configuration

| Parameter           | Value |
| ------------------- | ----- |
| Vocabulary Size     | 50257 |
| Context Length      | 256   |
| Embedding Dimension | 256   |
| Transformer Layers  | 12    |
| Attention Heads     | 4     |
| Dropout Rate        | 0.1   |

- #### Total Parameters: ~35.3M
- #### Context Window: 256 tokens

## Multi-head Attention

![Multi Head Attention](assets/multihead_attention.png)

## Model Architecture Diagram

![Model Architecture](assets/gpt_model.png)

### Data Sources

- English Language (Natural Language Understanding) : [TinyStories Train Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)

- Sarcasm Finetuning Dataset: [Sarcastic Tweets](https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony), AI Generated Question-Answer Synthetic Dataset
