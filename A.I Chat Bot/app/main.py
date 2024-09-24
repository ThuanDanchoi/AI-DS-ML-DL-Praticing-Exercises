from flask import Flask, request, jsonify
from model import load_model, generate_text
from mitigation import apply_mitigation

app = Flask(__name__)

# Load the model once when the app starts
model, tokenizer = load_model("gpt-neo")

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    # Generate initial response from the model
    response = generate_text(model, tokenizer, prompt)

    # Apply hallucination mitigation techniques
    mitigated_response = apply_mitigation(response)

    return jsonify({'response': mitigated_response})


if __name__ == '__main__':
    app.run(debug=True)






