from datasets import load_dataset
import openai
from openai import OpenAI
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
import os
import csv

train_ds2 = load_dataset("metaeval/reclor", split="validation")

def nl2cnf(text):
    # Split the text into sentences
    # sentences = sent_tokenize(text)
    sentences = sent_tokenize(text)
    # Build the single prompt
    prompt = (
        "You are an AI assistant that converts English sentences into Conjunctive normal form (CNF)."
        "Use SymPy's to_cnf() syntax, for example: P & Q, P | Q, ~P, ~P | Q, (P | ~Q) & (Q | ~P)."
        "If a statement repeats across multiple sentences, reuse the same variable. For example:\n"
        "\"John sings\" -> P\n"
        "\"If John sings, then Mary is happy.\" -> ~P | Q\n\n"
        "Represent each unique variable or statement using single alphabetic character, like A, B, C, D etc. Do not use compostion of words to represent variable.\n"
        "Do not use double alphabetic characters for one variable, only use number with alphabetic characters if all single alphabetic characters are used.\n"
        "Do not use symbols like ∧, ∨, or ->. Do not use triple backticks."
        "Connect individual clauses using & at output.\n\n" \
        "Simplify the final CNF expression as much as possible but do not use 'True' or 'False' as outpu, always use variables and CNF logical operators as output\n"
        "Output only the final simplified CNF expression, do not give any explaination.\n\n"
        "Here are the sentences:\n"
    )
    for i, sent in enumerate(sentences, start=1):
        prompt += f"{i}. {sent}\n"
    prompt += text 
    # Make a single request to the ChatGPT model (via "o1-mini" or whichever model)
    client = OpenAI()
    response = client.chat.completions.create(
        model="o3-mini",  # or "gpt-4", "o1-mini", etc.
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract the model's text response
    model_output = response.choices[0].message.content.strip()
    lines = model_output.splitlines()
    return [text, lines]

def remove_duplicate_lines(lines):
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
    return unique_lines

if __name__ == "__main__":
    # Make sure you have your OPENAI_API_KEY set
    openai.api_key = os.getenv("OPENAI_API_KEY")  # or directly: openai.api_key = "sk-..."

    # for i in range(0, 3):
    #     print(f"sample {i}\n")
    #     text = train_ds2[i]["context"]
    #     result = print_cnf(text)
    #     print(result[1])
    total_samples = len(train_ds2)
    with open('validation_cnf.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(['text', 'cnf'])

        for i in range(total_samples):
            print(f"Processing sample {i}/{total_samples}\n")
            text = train_ds2[i]["context"]
            result = nl2cnf(text)
            print(result[1])
            writer.writerow([result[0], result[1]])

            if i % 100 == 0:
                print(f"Processed {i}/{total_samples} samples")
            

        