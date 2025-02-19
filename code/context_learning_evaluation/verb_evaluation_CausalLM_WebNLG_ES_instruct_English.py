import os
import torch
import pandas as pd
import xml.etree.ElementTree as ET
import time
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

# Model list
models_list = ["Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "BSC-LT/salamandra-2b-instruct"]

access_token = "..."  # Ensure you have access

# Prompts
base_prompt = "In English, structured data is commonly represented as triples, with the format [subject, predicate, object]. Based on these triples, generate a single-paragraph text composed of complete, grammatically correct, and natural sentences. Generate the text solely from the following triples: \n"

# Format input for user message: providing the triples


one_shot_message = [{"role": "user", "content": (f"{base_prompt} Triples: \n" + 
    "[subject: 'Arion_(comicsCharacter)', predicate: 'creator', object: 'Jan_Duursema'] " +
    "[subject: 'Jan_Duursema', predicate: 'award', object: 'Eisner_Award'] " +
    "[subject: 'Arion_(comicsCharacter)', predicate: 'alternativeName', object: 'Ahri\\'ahn'] "+
    "[subject: 'Arion_(comicsCharacter)', predicate: 'creator', object: 'Paul_Kupperberg'] \n")},
    {"role": "system", "content": "Arion (also known as Ahri'ahn) is a comic character created by Paul Kupperberg and Jan Duursema, who won the Eisner award."}]

few_shot_message = [{"role": "user", "content": (f"{base_prompt} Triples: \n" + 
    "[subject: 'Arion_(comicsCharacter)', predicate: 'creator', object: 'Jan_Duursema'] " +
    "[subject: 'Jan_Duursema', predicate: 'award', object: 'Eisner_Award'] " +
    "[subject: 'Arion_(comicsCharacter)', predicate: 'alternativeName', object: 'Ahri\\'ahn'] "+
    "[subject: 'Arion_(comicsCharacter)', predicate: 'creator', object: 'Paul_Kupperberg'] \n")},
    {"role": "system", "content": "Arion (also known as Ahri'ahn) is a comic character created by Paul Kupperberg and Jan Duursema, who won the Eisner award."},
    {"role": "user", "content": (f"Triples: \n" + 
    "[subject: '11th_Mississippi_Infantry_Monument', predicate: 'category', object: 'Contributing_property'] "+
    "[subject: '11th_Mississippi_Infantry_Monument', predicate: 'municipality', object: 'Gettysburg,_Pennsylvania'] \n")},
    {"role": "system", "content": "The 11th Mississippi Infantry Monument is categorised as a contributing property and can be found in Gettysburg, Pennsylvania."}]


# Specify the quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True  # Set this to True for 8-bit or False for 4-bit
)

# Run models
for MODEL_NAME in models_list:
    print(f"Loading model {MODEL_NAME}...")
    file_model_name = MODEL_NAME.split("/")[1].replace('.', '')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=access_token, torch_dtype=torch.float16, device_map="auto")

    for mode, prompt, message, file_suffix in [("zero-shot", base_prompt, None, "zero_shot"), 
                                      ("one-shot", base_prompt, one_shot_message, "one_shot"), 
                                      ("few-shot", base_prompt, few_shot_message, "few_shot")]:
        print(f"Testing {MODEL_NAME} - {mode}...")

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
        output_path = f"./{file_model_name}_{file_suffix}_data.json"
        pd.DataFrame(results).to_json(output_path, orient="records", indent=4)
        print(f"Saved {output_path}!")

    del model  # Free memory
    torch.cuda.empty_cache()
