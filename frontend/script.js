// The backend and frontend are served from the SAME address (FastAPI serves
// this file too), so we can just call "/predict" directly.
const API_URL = "/predict";

async function predictNextWords() {
  const inputText = document.getElementById("inputText").value.trim();
  const numWords = parseInt(document.getElementById("numWords").value) || 1;

  const resultBox = document.getElementById("result");
  const errorBox = document.getElementById("error");
  const button = document.getElementById("predictBtn");

  errorBox.textContent = "";
  resultBox.textContent = "";

  if (!inputText) {
    errorBox.textContent = "Please type some text first.";
    return;
  }

  button.disabled = true;
  button.textContent = "Predicting...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: inputText, num_words: numWords }),
    });

    if (!response.ok) {
      throw new Error("Server returned an error");
    }

    const data = await response.json();
    resultBox.textContent = data.generated_text;
  } catch (err) {
    errorBox.textContent = "Could not reach the prediction server. Is app.py running?";
  } finally {
    button.disabled = false;
    button.textContent = "Predict Next Words";
  }
}
