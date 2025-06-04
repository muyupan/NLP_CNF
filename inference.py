import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

#model_path = "/storage/group/vxk1/default/muyu_folder/NLP_CNF/nl2prop_merged"
#model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
#tokenizer = AutoTokenizer.from_pretrained(model_path)

# Method 1: Fix the identifiers before pushing
#model.config._name_or_path = "muyu0515/nl2prop-deepseek8b"
#tokenizer.name_or_path = "muyu0515/nl2prop-deepseek8b"
#if hasattr(tokenizer, "tokenizer_config"):
#    tokenizer.tokenizer_config["name_or_path"] = "muyu0515/nl2prop-deepseek8b"

# Push to HuggingFace
#model.push_to_hub("muyu0515/nl2prop-deepseek8b", token=os.environ.get("HF_TOKEN"))
#tokenizer.push_to_hub("muyu0515/nl2prop-deepseek8b", token=os.environ.get("HF_TOKEN"))
from unsloth import FastLanguageModel
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')

# Load model from local path (change this to your actual path)
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
    dtype=None,
    load_in_4bit=True,
)

# Set to inference mode
FastLanguageModel.for_inference(model)

def generate_logic(text):
    sentences = sent_tokenize(text)
    
    prompt = (
        "You are an AI assistant that converts English sentences into logical expressions in propositional logic."
        "Use Python function-call syntax: And(P, Q), Or(P, Q), Not(P), Implies(P, Q), Equivalent(P, Q)."
        "If a statement repeats across multiple sentences, reuse the same variable. For example:\n"
        "\"John sings\" -> P\n"
        "\"If John sings, then Mary is happy.\" -> Implies(P, Q)\n\n"
        "Represent each unique variable or statement using single alphabetic character, like A, B, C, D etc. Do not use compostion of words to represent variable like Implies(ExecutiveBoardAppointment, UndergraduateDegree)!\n"
        "Do not use double alphabetic characters for one variable, only use number with alphabetic characters if all single alphabetic characters are used.\n"
        "Do not use symbols like ^, v, or ->. Do not use triple backticks."
        "Output only the function call syntax, one line per sentence. Do not explain sentence represented by what variables like 'Where:', 'P = In Country X last election, the Reform Party beat its main opponent, the Conservative Party. Only output propositional logic, do not do any explaination!\n\n"
        "Here are the sentences:\n"
    )
    
    # Add the numbered sentences
    for i, sent in enumerate(sentences, start=1):
        prompt += f"{sent}\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1)
    
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
