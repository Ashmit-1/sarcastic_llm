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



## Evaluation Questions and Answers:

### Question: What time is it now?
```
Answer:  Yeah, I'm just using a head to make up, because it's not nice to know the current
```
### Question: How's the weather today?
```
Answer:  Oh yeah, it's not real that people should get something else to be a big changes in the ground of a berry patch of all,
```
### Question: Did you finish the work?
```
Answer:  Yay, because the power of money from their pockets in the whole world and that the man was a better place of the last time to take
```
### Question: Is everything going well?
```
Answer:  But I don't know a movie, it's only for the people on the journey around by a problem but nobody would happen it, so I
```
### Question: Are you serious?
```
Answer:  I'm not sure what you can do when you're in the truth, it's not the most important place, I would be a bad about
```
### Question: Did you take your medicine?
```
Answer:  I am too strong if you want to buy a new game and have a good time with his body in his hand on it, so you can
```
### Question: Did you save the game?
```
Answer:  If the game's a game, we are the most happy times, the game is it all the way to the game and start a little one
```
### Question: Did you read the contract?
```
Answer:  No, they're so much better they can't read, but they can see that they're a big, white one on their hands of the
```
### Question: Is the device working properly?
```
Answer:  Yeah, a le of money to keep all the things in the right place so you have to do.

```
### Question: Did you check the settings?
```
Answer:  Yeah I'll do that we're right, I'm not being lazy, it's hard to do this right.
```
### Question: Did you call the doctor?
```
Answer:  Hey, it's ok," he said. "What do you think there is a problem for a lot of us."And they do.
```
### Question: Did you win the match?
```
Answer:  Yes, you don't want anything right and no season as well as a real baby's best of other baby, I don't want to be
```
### Question: Did you sign the document?
```
Answer:  The biggest person who have a lot of their original things like to do as much you can't be the person they had to be the same time
```
### Question: Is the battery charged?
```
Answer:  Yeah, I'm sure that the rules would get the right to your project.
```
### Question: Did you update the software?
```
Answer:  Y ** describes, and his once he goes for a turn before you get to have a chance to try this right before it'll be an instruction
```

