import os
import torch
import pandas as pd
import xml.etree.ElementTree as ET
import time
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

torch.cuda.empty_cache()

print(os.name, os.geteuid() if os.name != 'nt' else 'Admin' if os.system("net session >nul 2>&1") == 0 else 'Not Admin')

from transformers.utils import logging
logging.set_verbosity_error()  # Suppress warnings

# Function to parse a single XML file and extract data
def parse_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    entries_data = []

    for entry in root.findall(".//entry"):
        category = entry.attrib.get("category")
        size = entry.attrib.get("size")
        eid = entry.attrib.get("eid")

        english_triples = [format_triples(striple.text) for striple in entry.findall(".//modifiedtripleset/mtriple")]
        english_lex = [lex.text for lex in entry.findall(".//lex[@lang='en']")]

        entries_data.append({
            "category": category,
            "size": size,
            "eid": eid,
            "english_triples": english_triples,
            "english_lex": english_lex,
        })

    return entries_data

# Function to process XML files in a directory
def process_all_test_xml_files(directory_path):
    all_entries = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".xml"):
            all_entries.extend(parse_xml_file(os.path.join(directory_path, file_name)))

    return pd.DataFrame(all_entries)

def format_triples(triple_text):
    parts = triple_text.split(" | ")
    return {"subject": parts[0], "predicate": parts[1], "object": parts[2]}


from datetime import datetime
import torch

# Generate text from triples
def verbalize_triple(device, tokenizer, model, triples, prompt, messages = None):
    # Format input for user message: providing the triples
    if messages is None:
        messages = [{"role": "user", "content": f"{prompt} Triples: " + " ".join(
            f"[subject: '{t['subject']}', predicate: '{t['predicate']}', object: '{t['object']}'] " for t in triples
        )}]
    else:
        messages = messages + [{"role": "user", "content": f"Triples: " + " ".join(
            f"[subject: '{t['subject']}', predicate: '{t['predicate']}', object: '{t['object']}'] " for t in triples
        )}]
    
    # Format the prompt for chat-based instruction models
    date_string = datetime.today().strftime('%Y-%m-%d')
    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, date_string=date_string)
    
    # Tokenize the input for the model
    inputs = tokenizer.encode(prompt_formatted, add_special_tokens=False, return_tensors="pt").to(device)

    # Generate the text
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs, max_new_tokens=256)

    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the input part (prompt) to get only the generated text
    input_len = len(inputs[0])  # Get the length of the input tokens
    generated_text_without_input = generated_text[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]  # Remove input part

    return generated_text_without_input.strip()  # Only return the new generated text


# Load dataset
directory = "../WebNLG_ES/test/"
df_gt = process_all_test_xml_files(directory)
print("XML files loaded!")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(0)}")

# Define model path (change this to your saved model directory)

models = ["fine_tuned_qwen_0_5B", "fine_tuned_qwen_1_5B", "fine_tuned_llama_1B", "fine_tuned_salamandra_2B"]
for model_name in models:

    MODEL_PATH = f".../{model_name}/full_model"


    print("Loading model...")
    # Load the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")


    # Prompts
    base_prompt = "In English, structured data is commonly represented as triples, with the format [subject, predicate, object]. Based on these triples, generate a single-paragraph text composed of complete, grammatically correct, and natural sentences. Generate the text solely from the following triples: \n"

    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token

    for mode, prompt, message, file_suffix in [("zero-shot", base_prompt, None, "zero_shot")]:
        print(f"Testing fine tuned {MODEL_PATH} - {mode}...")

        results = []
        for _, row in tqdm(df_gt.iterrows(), total=len(df_gt)):
            start_time = time.time()
            generated = verbalize_triple(device, tokenizer, model, row["english_triples"], prompt, message)
            elapsed_time = time.time() - start_time

            results.append({
                'category': row["category"],
                'size': row["size"],
                'eid': row["eid"],
                'english_triples': row["english_triples"],
                'lex_gt': row["english_lex"],
                'prediction': generated,
                #'post_prediction': generated[1],
                'time': elapsed_time
            })

        # Save results
        output_path = f"./output_fine_tuned_english/{model_name}_Instruct_data.json"
        pd.DataFrame(results).to_json(output_path, orient="records", indent=4)
        print(f"Saved {output_path}!")

    del model  # Free memory
    torch.cuda.empty_cache()
