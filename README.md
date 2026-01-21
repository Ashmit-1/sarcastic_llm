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
