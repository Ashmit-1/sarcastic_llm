import torch
from untrained_model import GPTModel
import tiktoken
from untrained_model import generate
from untrained_model import text_to_token_ids
from untrained_model import token_ids_to_text
import os
import gdown


def format_input(entry):
    instruction_text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 
    
### Instruction:\n{entry['instruction']}
    """

    return instruction_text

def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 256,        # Embedding dimension
        "n_heads": 4,         # Number of attention heads
        "n_layers": 12,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }

    FILE_ID = "1AijrKkkfP25ujXl3Xe0wmzG8YaIkDwpP"
    if not os.path.exists("sarcasm_finetuned_v1.pth"):
        print("Downloading model from google drive... This may take some time.")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, "sarcasm_finetuned_v1.pth", quiet=False)
        print(f"Model downloaded !!")


    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("sarcasm_finetuned_v1.pth"))
    model.eval();

    try:
        tokenizer = tiktoken.get_encoding("gpt2")
    except Exception as e:
        print("Exception occured while tiktoken")



    inference_device = torch.device("cpu")

    model.to(inference_device)

    torch.manual_seed(123)
    model.to(inference_device)
    model.eval()

    input_query = input("Enter query: ")

    input_text = format_input({"instruction":input_query})

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(inference_device),
        max_new_tokens=30,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=10,
        temperature=0.9
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .replace("###", "")
        .replace("Response:", "")
        .replace("Response", "")
        .replace("<|endoftext|>", "")
        .strip()
    )
    print( response_text)





if __name__ == "__main__":
    main()
