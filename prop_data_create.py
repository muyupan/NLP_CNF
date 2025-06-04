from datasets import load_dataset
import openai
from openai import OpenAI
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
import os
import csv

# Load the dataset
train_ds2 = ds = load_dataset("metaeval/reclor", split="train")

def trans2prop(text):
    # Split the text into sentences
    sentences = sent_tokenize(text)
    # Build the single prompt
    prompt = (
        "You are an AI assistant that converts English sentences into logical expressions in propositional logic."
        "Use Python function-call syntax: And(P, Q), Or(P, Q), Not(P), Implies(P, Q), Equivalent(P, Q)."
        "If a statement repeats across multiple sentences, reuse the same variable. For example:\n"
        "\"John sings\" -> P\n"
        "\"If John sings, then Mary is happy.\" -> Implies(P, Q)\n\n"
        "Represent each unique variable or statement using single alphabetic character, like A, B, C, D etc. Do not use compostion of words to represent variable like Implies(ExecutiveBoardAppointment, UndergraduateDegree)!\n"
        "Do not use double alphabetic characters for one variable, only use number with alphabetic characters if all single alphabetic characters are used.\n"
        "Do not use symbols like ∧, ∨, or ->. Do not use triple backticks."
        "Output only the function call syntax, one line per sentence. Do not explain sentence represented by what variables like 'Where:', 'P = In Country X last election, the Reform Party beat its main opponent, the Conservative Party. Only output propositional logic, do not do any explaination!\n\n"
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
    openai.api_key = os.getenv("OPENAI_API_KEY")  
    total_samples = len(train_ds2)
    with open('train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text', 'propositional_logic'])

        for i in range(total_samples):
            print(f"Processing sample {i}/{total_samples}\n")
            text = train_ds2[i]["context"]
            result = trans2prop(text)
            result[1] = remove_duplicate_lines(result[1])
            print(result[1])
            writer.writerow([result[0], result[1]])

            if i % 100 == 0:
                print(f"Processed {i}/{total_samples} samples")
            

        