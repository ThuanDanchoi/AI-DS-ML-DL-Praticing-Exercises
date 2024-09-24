from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    """
    Load the pre-trained model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_text(model, tokenizer, prompt):
    """
    Generate text from the model based on the given prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
