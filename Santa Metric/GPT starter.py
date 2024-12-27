# Import necessary modules
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize the model and tokenizer
model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)

def calculate_perplexity(sentence):
    """Calculate perplexity for a given sentence."""
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

# Calculate perplexity for each permutation
perplexities = [(sentence, calculate_perplexity(sentence)) for sentence in permuted_sentences]

# Find the sentence with the lowest perplexity
best_sentence, best_perplexity = min(perplexities, key=lambda x: x[1])

# Display the result
best_sentence, best_perplexity


##---------------------------------------------------------------
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b")

import os
os.system("huggingface-cli login")