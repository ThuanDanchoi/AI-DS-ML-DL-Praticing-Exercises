from flask import Flask, render_template, request
from paraphrasing_model import paraphrase_text

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('index.html')


@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    input_text = request.form['input_text']
    paraphrased_output = paraphrase_text(input_text)  # Generate paraphrased text

    # Debugging print
    print(f"Paraphrased Output: {paraphrased_output}")  # Check what is being returned

    return render_template('index.html', original=input_text, paraphrased=paraphrased_output)


if __name__ == '__main__':
    app.run(debug=True)
