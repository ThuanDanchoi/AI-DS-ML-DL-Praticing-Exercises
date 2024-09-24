from flask import Flask, render_template, request
from model import load_model, generate_text
from mitigation import apply_mitigation

app = Flask(__name__)

# Load the model and tokenizer
model, tokenizer = load_model("gpt-neo")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    prompt = request.form['prompt']
    if prompt:
        response = generate_text(model, tokenizer, prompt)
        mitigated_response = apply_mitigation(response)
        return render_template('index.html', prompt=prompt, response=mitigated_response)
    else:
        return render_template('index.html', error='Please enter a prompt.')

if __name__ == '__main__':
    app.run(debug=True)
