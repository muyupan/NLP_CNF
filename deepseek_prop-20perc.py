from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
import time
import json
from datetime import datetime

#1. load a model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit", 
    dtype=None,
    load_in_4bit=True,
)

#2. set chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama3",
)

#3. Tokenization using chatml
def formatting_prompts_func(examples):
    conversations = []
    
    base_prompt = (
        "You are an AI assistant that converts English sentences into logical expressions in propositional logic."
        "Here are the sentences:\n"
    )
    full_prompt = base_prompt
    for text, logic in zip(examples["Text"], examples["Propositional Logic"]):
        sentences = sent_tokenize(text)
        
        for i, sent in enumerate(sentences, start=1):
            full_prompt += f"{sent}\n"
        
        convo = [
            {"role": "user", "content": full_prompt},
            {"role": "assistant", "content": logic}
        ]
        conversations.append(convo)

    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
             for convo in conversations]

    return {"text": texts}

#load dataset
dataset = load_dataset("muyu0515/nl2prop", split={
    "train": "train[:10%]",
    "validation": "validation",
    "test": "test"
})

dataset["train"] = dataset["train"].map(formatting_prompts_func, batched=True)
dataset["test"] = dataset["test"].map(formatting_prompts_func, batched=True)
dataset["validation"] = dataset["validation"].map(formatting_prompts_func, batched=True)
#4. LoRA setting
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",  
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False, 
    loftq_config = None,
)

#5. Set Training Params
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=4,
    learning_rate=2e-4,
    warmup_steps=100,

    # Validation settings
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,  
    metric_for_best_model="eval_loss",

    output_dir="./nl2prop_deepseek8b",
    logging_steps=20,
    report_to="none",  # Disable wandb if not using
)

#6. Create Trainer and Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    args=training_args,
)

start_time = time.time()
start_datetime = datetime.now()

trainer.train()

end_time = time.time()
training_hours = (end_time - start_time) / 3600
print(training_hours)
#7. Save and exportt
model.save_pretrained("nl2prop_deepseek8b")
tokenizer.save_pretrained("nl2prop_deepseek8b")
model.save_pretrained_merged("nl2prop_merged", tokenizer, save_method="merged_16bit")

#model.save_pretrained_gguf(
#    "nl2prop_deepseek8b",
#    tokenizer,
#    quantization_method="q4_k_m"
#)

#model.push_to_hub_gguf(
#    "muyu0515/nl2prop_deepseek8b", 
#    tokenizer,
#    quantization_method="q4_k_m",
#    private=True
#)

#8. Inference
FastLanguageModel.for_inference(model)
def generate_logic(text):
    sentences = sent_tokenize(text)
    
    prompt = (
        "You are an AI assistant that converts English sentences into logical expressions in propositional logic."
        "Here are the sentences:\n"
    )
    
    for i, sent in enumerate(sentences, start=1):
        prompt += f"{sent}\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

result = generate_logic("Peter eats and Lora sleeps.")
print(result)
