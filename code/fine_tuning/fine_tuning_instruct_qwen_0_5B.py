import os
import torch
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import Trainer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset, DatasetDict
from accelerate import Accelerator

# Empty cache before starting
torch.cuda.empty_cache()

# ------------------ CONFIGURATION ------------------
access_token = "..."  # Ensure you have access
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 
OUTPUT_DIR = "./fine_tuned_qwen2_5_0_5B_Instruct" # Change to path in your directory
base_prompt = ("En español, los datos estructurados se representan comúnmente como tripletas o triples, "
               "con el formato [sujeto, predicado, objeto]. A partir de estas tripletas, genera un texto de un solo párrafo "
               "formado por oraciones completas, gramaticalmente correctas y naturales. Genera el texto únicamente a partir "
               "de las siguientes tripletas: \n")

# ------------------ ACCELERATE ------------------
accelerator = Accelerator()

# LoRA Configuration (Lightweight adapter)
lora_config = LoraConfig(
    r=4,  # Reduce rank for faster training
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ------------------ DATA LOADING ------------------
def load_data(split):
    """Load and format the dataset for training."""
    xml_dir = f"../WebNLG_ES/{split}/"
    data = []

    # Recursively explore all subdirectories and load XML files
    for root_dir, _, files in os.walk(xml_dir):
        for xml_file in files:
            if xml_file.endswith(".xml"):
                file_path = os.path.join(root_dir, xml_file)
                tree = ET.parse(file_path)
                root = tree.getroot()

                for entry in root.findall(".//entry"):
                    # Extract all triples from spanishtripleset
                    triples = [triple.text.strip() for triple in entry.findall(".//spanishtripleset/striple")]
                    # Extract the corresponding text in Spanish from the lex elements
                    texts = [lex.text.strip() for lex in entry.findall(".//lex[@lang='es']")]

                    # Only proceed if we have at least one lex entry
                    if texts:
                        # Combine all triples into one user message
                        combined_triples = " ".join([f"[sujeto: '{triple.split(' | ')[0]}', predicado: '{triple.split(' | ')[1]}', objeto: '{triple.split(' | ')[2]}']" for triple in triples])
                        prompt = f"{base_prompt} Tripletas: \n{combined_triples} \n"
                        response = texts[0]  # Take the first lex entry in Spanish
                        
                        # Format the data as one_shot_message format
                        one_shot_message = [
                            {"role": "user", "content": prompt},
                            {"role": "system", "content": response}
                        ]
                        data.append({"text": one_shot_message})

    dataset = Dataset.from_list(data)

    #print(dataset)

    # Tokenize the dataset using apply_chat_template
    def tokenize_function(examples):
        # Apply the tokenizer's chat template for each example
        chat_input = tokenizer.apply_chat_template(examples['text'], tokenize=False, add_generation_prompt=True)

        # Tokenize inputs efficiently
        tokenization = tokenizer(chat_input, padding="max_length", truncation=True, max_length=512)  # Set a max length to avoid long inputs

        return tokenization
    
    # Apply tokenization to the dataset
    return dataset.map(tokenize_function, batched=False)

# ------------------ MODEL LOADING ------------------
# Load model with mixed precision and offloading (device_map="auto" for large models)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=access_token, torch_dtype=torch.float16, device_map="auto")
print(model)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Function to calculate and print the number of trainable parameters
def print_trainable_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {total_params - trainable_params}")

# Apply LoRA adapters for efficient training
model = get_peft_model(model, lora_config)

# After loading the model with PEFT (LoRA)
print_trainable_params(model)

# ------------------ TRAINING ------------------
train_dataset = load_data("train")
dev_dataset = load_data("dev")

dataset = DatasetDict({
    "train": train_dataset,
    "validation": dev_dataset
})

print(dataset)

torch.cuda.empty_cache()

# Adjust Training Arguments for memory efficiency
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,  # Small batch size
    per_device_eval_batch_size=1,  # Small batch size
    gradient_accumulation_steps=8,  # Accumulate gradients to simulate a larger batch
    eval_strategy="steps",
    eval_steps=250,  # More frequent evaluation for faster feedback
    save_strategy="steps",
    save_steps=250,  # Save model checkpoints frequently
    logging_steps=25,
    learning_rate=1e-4,  # Adjust learning rate for faster convergence
    weight_decay=0.01,
    fp16=True,  # Mixed Precision Training (FP16)
    num_train_epochs=2,  # Reduce epochs for quicker fine-tuning
    push_to_hub=False,
    save_total_limit=2,
    optim="adamw_torch",  # Use standard AdamW optimizer
    report_to="none",  # Disable reporting
)

# Use the trainer for model training
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_args,
    processing_class=tokenizer  # Correct argument (tokenizer -> processing_class)
)

torch.cuda.empty_cache()

trainer.train()

# ------------------ SAVE MODELS ------------------

# Save adapter-only model (LoRA weights)
adapter_output_dir = os.path.join(OUTPUT_DIR, "adapter_model")
model.save_pretrained(adapter_output_dir)
tokenizer.save_pretrained(adapter_output_dir)

# Save full model (merge LoRA with base model and then save)
full_model_output_dir = os.path.join(OUTPUT_DIR, "full_model")
full_model = model.merge_and_unload()  # Merge LoRA adapters with the base model
full_model.save_pretrained(full_model_output_dir)
tokenizer.save_pretrained(full_model_output_dir)
