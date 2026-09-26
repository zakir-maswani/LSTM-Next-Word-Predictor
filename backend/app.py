import json
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
import nltk

from model import LSTMModel

# Load the vocab, config and trained model (this runs once, when server starts)
with open(r"C:\Users\TAQICOMPUTERS\Desktop\lstm_next_word_predictor\saved_model\vocab.json") as f:
    vocab = json.load(f)

with open(r"C:\Users\TAQICOMPUTERS\Desktop\lstm_next_word_predictor\saved_model\config.json") as f:
    config = json.load(f)

# index -> word, so we can turn a predicted number back into a word
index_to_word = {index: word for word, index in vocab.items()}

INPUT_LEN = config["input_len"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(vocab_size=config["vocab_size"])
model.load_state_dict(torch.load(r"C:\Users\TAQICOMPUTERS\Desktop\lstm_next_word_predictor\saved_model\model.pth", map_location=device))
model.to(device)
model.eval()

# Helper functions (same logic as the notebook's `prediction` function)
def text_to_indices(sentence_tokens):
    return [vocab.get(token, vocab["<unk>"]) for token in sentence_tokens]


def predict_next_word(text):
    """Given some text, predict the single next word."""
    tokens = word_tokenize(text.lower())
    numerical_text = text_to_indices(tokens)

    # pad (or trim) so the input is exactly INPUT_LEN long, like during training
    if len(numerical_text) < INPUT_LEN:
        numerical_text = [0] * (INPUT_LEN - len(numerical_text)) + numerical_text
    else:
        numerical_text = numerical_text[-INPUT_LEN:]

    padded_text = torch.tensor(numerical_text, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(padded_text)
        _, predicted_index = torch.max(output, dim=1)

    return index_to_word.get(predicted_index.item(), "<unk>")


def predict_next_words(text, num_words):
    """Predict several words in a row by feeding each prediction back in."""
    result_text = text
    for _ in range(num_words):
        next_word = predict_next_word(result_text)
        result_text = result_text + " " + next_word
    return result_text

# FastAPI app
app = FastAPI(title="LSTM Next Word Predictor")

# allow the frontend (any origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str
    num_words: int = 1   # how many next words to predict


@app.post("/predict")
def predict(request: PredictRequest):
    num_words = max(1, min(request.num_words, 20))  # keep it reasonable
    generated_text = predict_next_words(request.text, num_words)
    return {
        "input_text": request.text,
        "generated_text": generated_text,
    }


@app.get("/info")
def info():
    return {
        "vocab_size": config["vocab_size"],
        "accuracy": config["accuracy"],
    }


# Serve the frontend files directly from FastAPI, so you only need
# one server running (no separate frontend server needed).
app.mount("/", StaticFiles(directory=r"C:\Users\TAQICOMPUTERS\Desktop\lstm_next_word_predictor\frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)