# LSTM Next Word Predictor

A simple **Next Word Prediction** project built with **PyTorch (LSTM)**, served through a **FastAPI** backend, with a plain **HTML/CSS/JavaScript** frontend.

Type some text in the browser and the trained LSTM model predicts the next word(s) — for real, using the actual trained model (not fake/random text).

---

## 📁 Project Structure

```
lstm_next_word_predictor/
│
├── notebook/
│   └── lstm_next_word_predictor.ipynb   # Organized Jupyter notebook (all steps, with headings)
│
├── backend/
│   ├── data.py            # The training text (document)
│   ├── model.py            # LSTM model definition (shared by train.py and app.py)
│   ├── train.py            # Preprocesses data, trains model, checks accuracy, saves model
│   └── app.py               # FastAPI server -> loads saved model, serves predictions + frontend
│
├── frontend/
│   ├── index.html          # Simple UI
│   ├── style.css           # Styling
│   └── script.js            # Calls the FastAPI /predict endpoint
│
├── saved_model/            # Created after running train.py
│   ├── model.pth            # Trained model weights
│   ├── vocab.json           # Word -> index mapping
│   └── config.json          # vocab size, sequence length, accuracy
│
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works (Steps Followed)

Same steps as the original notebook, just organized and documented:

1. **Tokenize** the text document (using NLTK).
2. **Build vocabulary** — map every unique word to a number.
3. **Build training sequences** — for every sentence, create growing sub-sequences (e.g. `[the, course] -> fee`).
4. **Pad** all sequences to the same length.
5. **Create Dataset & DataLoader** (PyTorch).
6. **Define the LSTM model** — Embedding → LSTM → Linear layer.
7. **Train** the model for 50 epochs.
8. **Check accuracy** on the training data.
9. **Save** the model weights + vocabulary + config so the FastAPI backend can use them.

---

## ▶️ How to Run

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Train the model (this creates the `saved_model/` files)

```bash
cd backend
python train.py
```

You will see the training loss printed for 50 epochs, followed by the model's accuracy, and finally a confirmation that `model.pth`, `vocab.json` and `config.json` were saved.

### Step 3 — Start the FastAPI server

Still inside the `backend/` folder:

```bash
uvicorn app:app --reload
```

### Step 4 — Open the app

Open your browser at:

```
http://127.0.0.1:8000
```

You'll see a simple page — type some starting text, choose how many words to predict, and click **"Predict Next Words"**. The prediction comes from the real trained LSTM model through the `/predict` API endpoint.

---

## 🔌 API Endpoints

| Method | Endpoint   | Description                                  |
|--------|-----------|-----------------------------------------------|
| POST   | `/predict` | Body: `{ "text": "...", "num_words": 3 }` → returns generated text |
| GET    | `/info`    | Returns vocab size and training accuracy      |

---

## 📝 Notes

- The training text used here (`backend/data.py`) is a fictional FAQ about a "Full Stack Web Development Bootcamp" — you can replace it with your own text; just re-run `train.py` afterwards to retrain the model on the new text.
- This is a small, from-scratch model trained on a small custom dataset, so predictions work best for text similar in style to the training document. It's meant as a learning project, not a production-grade language model.
- If you change the training text, delete the old files in `saved_model/` and run `train.py` again before starting the server.
