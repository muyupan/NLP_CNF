from datasets import load_dataset
import openai
from openai import OpenAI
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
import os

train_ds2 = ds = load_dataset("metaeval/reclor", split="train")

def print_prop(text):
    # Split the text into sentences
    # sentences = sent_tokenize(text)

    # Build the single prompt
    prompt = (
        "You are an AI assistant that converts English sentences into logical expressions in propositional logic. "
        "Use Python function-call syntax: And(P, Q), Or(P, Q), Not(P), Implies(P, Q), Equivalent(P, Q). "
        "If a statement repeats across multiple sentences, reuse the same variable. For example:\n"
        "\"John sings\" -> P\n"
        "\"If John sings, then Mary is happy.\" -> Implies(P, Q)\n\n"
        "Do not use symbols like ∧, ∨, or ->. Do not use triple backticks. "
        "Output only the function call syntax, one line per sentence.\n\n"
        "Here are the sentences:\n"
    )
    # for i, sent in enumerate(sentences, start=1):
    #     prompt += f"{i}. {sent}\n"
    prompt += text 
    # Make a single request to the ChatGPT model (via "o1-mini" or whichever model)
    client = OpenAI()
    response = client.chat.completions.create(
        model="o3-mini",  # or "gpt-4", "o1-mini", etc.
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract the model's text response
    model_output = response.choices[0].message.content.strip()
    print(f"text: {text}")
    print(f"Model Output: \n{model_output}")

    # lines = model_output.splitlines()

    # # Parse each line 
    # for i, (sentence, line) in enumerate(zip(sentences, lines), start=1):
    #     line_stripped = line.strip()
    #     if not line_stripped:
    #         continue

    #     print(f"Sentence #{i}: {sentence}")
    #     print(f"Model Output: {line_stripped}")

if __name__ == "__main__":
    # Make sure you have your OPENAI_API_KEY set
    openai.api_key = os.getenv("OPENAI_API_KEY")  # or directly: openai.api_key = "sk-..."

    for i in range(0, 3):
        print(f"sample {i}\n")
        text = train_ds2[i]["context"]
        print_prop(text)
        
from datasets import load_dataset
import shutil
import os

ds = load_dataset("muyu0515/nl2prop")

print(ds["train"][0])