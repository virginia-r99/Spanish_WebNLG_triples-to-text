import os
import json
import torch
from sacrebleu import corpus_bleu, corpus_chrf, corpus_ter
from nltk.translate.meteor_score import meteor_score
import spacy  # Import spaCy for Spanish tokenization
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from evaluate import load  # Import Hugging Face's evaluate library

# Load BERTScore from evaluate
bertscore = load("bertscore")

print("Loading spacy model...")
# Load Spanish spaCy model
nlp = spacy.load("es_core_news_lg")

def preprocess_text(text):
    """Preprocess text: word-level tokenization, lowercasing, and punctuation removal."""
    doc = nlp(text.lower())  # Convert to lowercase to standardize
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    return tokens

def compute_metrics(references, prediction):
    """Computes lexical evaluation metrics."""
    if not references:
        return {"BLEU": 0, "METEOR": 0, "CHRF++": 0, "TER": 100}
    
    reference_tokens = [preprocess_text(ref) for ref in references]
    prediction_tokens = preprocess_text(prediction)
    
    prediction_text = " ".join(prediction_tokens)
    reference_texts = [" ".join(ref_tokens) for ref_tokens in reference_tokens]

    bleu = max(corpus_bleu([prediction_text], [[ref_text]]).score for ref_text in reference_texts)
    chrf = max(corpus_chrf([prediction_text], [[ref_text]], beta=2).score for ref_text in reference_texts)
    ter = max(corpus_ter([prediction_text], [[ref_text]]).score for ref_text in reference_texts)
    meteor = max(meteor_score([ref_tokens], prediction_tokens) for ref_tokens in reference_tokens)

    return {"BLEU": bleu, "METEOR": meteor, "CHRF++": chrf, "TER": ter}

def compute_embedding_similarity(model, references, prediction):
    """Computes embedding-based cosine similarity using Sentence Transformers."""
    if not references:
        return 0.0
    
    ref_embeddings = model.encode(references, convert_to_tensor=True)
    pred_embedding = model.encode(prediction, convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(pred_embedding, ref_embeddings)
    return float(torch.max(cosine_similarities).item())

def compute_bertscore(references, prediction):
    """Computes BERTScore for semantic similarity, taking the best F1 score."""
    if not references:
        return {"BERTScore_Precision": 0, "BERTScore_Recall": 0, "BERTScore_F1": 0}

    # Wrap references in a list of lists to ensure correct input format
    results = bertscore.compute(predictions=[prediction], references=[references], lang="es")

    # Get the best F1 score and corresponding precision and recall
    best_f1_index = max(range(len(results["f1"])), key=lambda i: results["f1"][i])

    return {
        "BERTScore_Precision": results["precision"][best_f1_index],
        "BERTScore_Recall": results["recall"][best_f1_index],
        "BERTScore_F1": results["f1"][best_f1_index]
    }


def process_json_file(filename, folder_path, output_folder_path, model):
    file_path = os.path.join(folder_path, filename)
    file_output_path = os.path.join(output_folder_path, filename)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for entry in tqdm(data):
        lex_gt = entry.get("lex_gt", [])
        prediction = entry.get("prediction", "").strip()
        
        if lex_gt and prediction:
            entry["eval"] = compute_metrics(lex_gt, prediction)
            entry["eval"]["Cosine_Similarity"] = compute_embedding_similarity(model, lex_gt, prediction)
            entry["eval"].update(compute_bertscore(lex_gt, prediction))
    
    output_file_path = file_output_path.replace(".json", "_evaluate.json")
    
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# Define the folders
test = "test_1"
folder_path = f"../../results/context_learning/{test}"  # Change this to your folder path
output_folder_path = f"../../results/context_learning/{test}"

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  

# Save evaluation results
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  # Ensure it's a JSON file
        print(filename)
        process_json_file(filename, folder_path, output_folder_path, model)