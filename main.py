import torch
from setup_pretrainning.untrained_model import GPTModel
import tiktoken
from setup_pretrainning.untrained_model import generate
from setup_pretrainning.untrained_model import text_to_token_ids
from setup_pretrainning.untrained_model import token_ids_to_text
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


    if not os.path.exists("sarcasm_finetuned_v2.pth"):
        print("Downloading model from google drive... This may take some time.")
        # url = "https://drive.google.com/file/d/1AijrKkkfP25ujXl3Xe0wmzG8YaIkDwpP/view?usp=sharing"
        url = "https://drive.google.com/file/d/1SGX9HtMldDOgGcwM2QBI7DSRCkYDe3X/view?usp=sharing"
        
        gdown.download(url, "sarcasm_finetuned_v1.pth", quiet=False)
        print(f"Model downloaded !!")


    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("sarcasm_finetuned_v2.pth"))
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

    # input_query = input("Enter query: ")

    query_list = [
        "What time is it now?",
        "How's the weather today?",
        "Did you finish the work?",
        "Is everything going well?",
        "Are you serious?",
        "Did you take your medicine?",
        "Did you save the game?",
        "Did you read the contract?",
        "Is the device working properly?",
        "Did you check the settings?",
        "Did you call the doctor?",
        "Did you win the match?",
        "Did you sign the document?",
        "Is the battery charged?",
        "Did you update the software?"
    ]
    for input_query in query_list:

        input_text = format_input({"instruction":input_query})

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(inference_device),
            max_new_tokens=35,
            context_size=GPT_CONFIG_124M["context_length"],
            top_k=25,
            temperature=0.95
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
        print(f"### Question: {input_query}")
        print("```")
        print("Answer: ", response_text)
        print("```")






if __name__ == "__main__":
    main()
