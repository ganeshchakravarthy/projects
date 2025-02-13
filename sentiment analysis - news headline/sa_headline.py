from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

sa_pipe = pipeline("text-classification", model="ProsusAI/finbert")


@app.route('/', methods= ['POST'])

def sa():
    data = request.get_json()
    headline = data.get("Enter the headline:", "")

    if not headline:
        return jsonify({"error: No headline is given."}), 400

    result = sa_pipe(headline)

    sentiment = result[0]['label']
    score = result[0]['label']

    return jsonify({"headline": headline, "sentiment": sentiment, "confidence" : round(score, 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
