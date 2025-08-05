import torch
import csv
import os
from transformers import AutoTokenizer, AutoModel

# BioBERT and tokenizer load
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# check CUDA 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if torch.cuda.is_available():
    print(f"CUDA ok. using GPU.")
else:
    print("CUDA ok. using CPU.")

# 512 tokens to a chunk
def chunk_text(text, max_tokens=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks

# chunk to embedding
def get_biobert_embeddings(text):
    embeddings = []
    for chunk in chunk_text(text):
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # [CLS]
        embeddings.append(cls_embedding.cpu())
    final_embedding = torch.cat(embeddings, dim=0)
    return final_embedding

output_dir = "./biobert_embeddings"
os.makedirs(output_dir, exist_ok=True)

# merge drug name and description
with open('drugbank_descriptions.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    for row in reader:
        drug_id = row['DrugBank ID']
        drug_name = row['Name'] if row['Name'] else "Unknown"
        drug_description = row['Description'] if row['Description'] else ""

        if not drug_description.strip():
            print(f"{drug_id} description doesn't exits.")
            continue

        # check for drug name in description
        if drug_name not in drug_description:
            combined_text = f"DrugName: {drug_name}. Description: {drug_description}"
        else:
            combined_text = drug_description

        try:
            embedding = get_biobert_embeddings(combined_text)
            save_path = os.path.join(output_dir, f"{drug_id}.pt")
            torch.save(embedding, save_path)
            print(f"{drug_id} embedding saved:", save_path)
        except Exception as e:
            print(f"Error processing {drug_id}: {e}")
            continue

print("Making BioBERT embeddings completed.")

