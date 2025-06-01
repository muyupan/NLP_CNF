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

# Now load from HuggingFace with Unsloth
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "muyu0515/nl2prop-deepseek8b",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Test inference
inputs = tokenizer(
    "If it's sunny, we go to the beach.",
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
