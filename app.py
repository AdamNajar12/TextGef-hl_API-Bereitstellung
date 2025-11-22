import os
from flask import Flask, request, jsonify
from transformers import pipeline

# ✅ Set writable cache for HuggingFace transformers
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache"
os.makedirs("/tmp/.cache", exist_ok=True)

app = Flask(__name__)
application = app

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")

print(f"Loading model {MODEL_NAME} ...")
classifier = pipeline("sentiment-analysis", model=MODEL_NAME)
print("Model loaded.")

@app.route("/")
def home():
    return jsonify({"message": "Model API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text") or request.args.get("text")
    if not text:
        return jsonify({"error": "please provide 'text' in JSON body or ?text=..."}), 400
    out = classifier(text)
    return jsonify({"result": out})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
