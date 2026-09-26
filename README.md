# LSTM Next Word Predictor

A simple **Next Word Prediction** project built with **PyTorch (LSTM)**, served through a **FastAPI** backend, with a plain **HTML/CSS/JavaScript** frontend.

Type some text in the browser and the trained LSTM model predicts the next word(s) — for real, using the actual trained model (not fake/random text).

---

## 📁 Project Structure

```
(Updated soon)

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

```
Updated soon
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

