# Fine-Tuning Guide for the Provided Model (.pth)

This guide explains how to fine-tune the provided `.pth` model using the supplied training script.

The script:
- Loads a pretrained checkpoint (`model_and_optimizer_small.pth` --> this will not be the model name in your case so you need to change the file name to the name of the .pth file in your case)
- Fine-tunes it on instruction–response data
- Saves the fine-tuned model as a new `.pth` file

---

#  Requirements

## Environment

Make sure you have:

- Python 3.9+
- PyTorch (>= 2.0 recommended)
- CUDA (optional but recommended for GPU training)
- Your tokenizer (same one used to train the base model)
- `untrained_model.py` file (contains `GPTModel`, `generate`, etc.)

Install PyTorch:

```bash
pip install torch
```

If using GPU, install the CUDA-enabled PyTorch version from:
https://pytorch.org/get-started/locally/

---

#  Folder Structure

Your project should look like this (this structure only shows the files required for finetuning):

```
project_root/
│
├── dataset/
│   └── sarcasm/
│       └── sarcasm_instruction_pairs2.jsonl
├── setup_pretraining/
│    └── finetuning.ipynb
│    
├── model_and_optimizer_small.pth

```

- `model_and_optimizer_small.pth` → Base checkpoint (model + optimizer)
- `finetuning.ipynb` → The fine-tuning script
- `dataset/...jsonl` → Training data

---

#  Dataset Format (VERY IMPORTANT)

Your data **must** be in `.jsonl` format (one JSON per line).

Each line must contain:

```json
{
  "instruction": "Your instruction text",
  "response": "Expected response text"
}
```

### Example

```json
{"instruction": "What do you think about the education systems in Sweden and Japan?", "response": "In advanced economies such as Sweden and Japan, 3yo kids learn how to solve 2 unknown equations already"}
{"instruction": "What do you think about McDormett's potential in basketball?", "response": "I honestly think McDormett has the potential to be the white Larry Bird if taught by the right people."}
```

---

#  How the Model Sees the Data

The script formats each example like this:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
<instruction text>

### Response:
<response text>
```

So for any new domain (medical, coding, law, etc.), just change the instruction/response content — the format must remain consistent.

---

#  Train / Validation / Test Split

The script automatically splits the dataset:

- 85% → Training
- 10% → Testing
- 5% → Validation

No manual splitting is required.

---

#  Model Configuration

The script loads the base model using:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

⚠️ Do **NOT** change these values unless you also change the base model architecture.

They must match the original pretrained model.

---

#  Loading the Base Model

The script loads:

```python
checkpoint = torch.load("model_and_optimizer_small.pth", weights_only=True) #change this to the model file name for your case
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

So your base file **must contain**:

- `model_state_dict`
- `optimizer_state_dict`

---

#  Training Hyperparameters

Current settings:

```python
batch_size = 8
learning_rate = 5e-5
weight_decay = 0.1
num_epochs = 4
max_sequence_length = 1024
```

You may adjust:

- `batch_size` (lower if GPU runs out of memory)
- `num_epochs` (increase for more training)
- `learning_rate` (lower if unstable)

---

#  Running the Training

- To use this script you need to change the model file name to proper .pth file in your case.
- Make sure the finetuning data is in jsonl format.
- The finetuing data followes the "instruction" and "response" order

---

#  Saving the Fine-Tuned Model

After training, the script saves:

```python
sarcasm_finetuned_v2.pth
```

This file contains:

```python
model.state_dict()
```

To load it later:

```python
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("sarcasm_finetuned_v2.pth"))
model.eval()
```

---

#  Changes you need to do to finetune the model

To fine-tune :

### Step 1 — Create new dataset

Example: medical Q&A

```json
{"instruction": "What are the symptoms of dehydration?", "response": "Common symptoms include dry mouth, fatigue, dizziness and dark urine."}
```

### Step 2 — Replace dataset path

In the script:

```python
file_path = "dataset/your_domain/your_file.jsonl"
```
### Step 3 — Replace the model path

In the script:

```python
checkpoint = torch.load("name_of_model_in_your_case.pth", weights_only=True)
```

### Step 4 — Run training

That's it.

No other code changes required.

---


#  Final Checklist Before Training

✅ Correct `.jsonl` format  
✅ Correct tokenizer  
✅ Correct base checkpoint  
✅ Matching model config  
✅ Proper dataset path  
✅ Enough GPU memory  

---

# To test the newly finetuned model

- All the steps to get inference from the newly finetuned model are specified in the README file

- You just need to change the file name in the main.py file to the newly created .pth file after finetuning

- Using the main.py script you can now get responses from the new model.
---

# Done 


Happy fine-tuning !!
