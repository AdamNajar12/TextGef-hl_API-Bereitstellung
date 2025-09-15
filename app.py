from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)
# S2I/WSGI expects `application` variable
application = app

# pick model via env var or default
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")

# load model at startup (this can be slow because it downloads weights)
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

# ✅ Add this block for Windows local testing
if __name__ == "__main__":
    # host=0.0.0.0 makes it reachable from localhost and OpenShift style
    app.run(host="0.0.0.0", port=8080, debug=True)
