from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')


def paraphrase_text(text, max_length=256):
    input_text = "paraphrase: " + text + " </s>"
    encoding = tokenizer.encode_plus(input_text, return_tensors="pt", max_length=max_length, truncation=True)

    # Generate paraphrased output
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            max_length=max_length,
            num_beams=5,
            num_return_sequences=1,
            repetition_penalty=2.5,
            length_penalty=1.0
        )

    # Decode and return the output
    paraphrased_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Double check if it's returning the correct string
    return paraphrased_text
