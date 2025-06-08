import os
os.environ['HF_HOME'] = '/storage/group/vxk1/default/muyu_folder/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/storage/group/vxk1/default/muyu_folder/.cache/huggingface/transformers'
os.environ['HF_HUB_CACHE'] = '/storage/group/vxk1/default/muyu_folder/.cache/huggingface/hub'
nltk_dir = '/storage/group/vxk1/default/muyu_folder/nltk_data'
os.makedirs(nltk_dir, exist_ok=True)

from unsloth import FastLanguageModel
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', download_dir=nltk_dir)
nltk.download('punkt_tab', download_dir=nltk_dir)
# Load model from local path (change this to your actual path)
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    dtype=None,
    load_in_4bit=True,
)

# Set to inference mode
FastLanguageModel.for_inference(model)

def generate_logic(text):
    sentences = sent_tokenize(text)
    
    prompt = (
        "You are an AI assistant that converts English sentences into logical expressions in propositional logic."
        "Here are the sentences:\n"
    )
    
    # Add the numbered sentences
    for i, sent in enumerate(sentences, start=1):
        prompt += f"{sent}\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, temperature=0.1)
    
    # Extract just the generated part
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from output
    result = generated[len(prompt):].strip()
    
    return result

# Test with different examples
test_cases = [
    'In rheumatoid arthritis, the body s immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints.'
]

for test in test_cases:
    print(f"Input: {test}")
    print(f"Output: {generate_logic(test)}")
    print("-" * 50)# Now load from HuggingFace with Unsloth
