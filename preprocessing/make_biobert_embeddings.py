import torch
import csv
import os
from transformers import AutoTokenizer, AutoModel

# Load BioBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("CUDA ok. using", "GPU." if torch.cuda.is_available() else "CPU.")

# Split long text into chunks of <= 512 tokens
def chunk_text(text, max_tokens=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks

# Convert text to BioBERT embedding
def get_biobert_embeddings(text):
    embeddings = []
    for chunk in chunk_text(text):
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # Use [CLS] token
        embeddings.append(cls_embedding.cpu())
    final_embedding = torch.cat(embeddings, dim=0)
    return final_embedding

output_dir = "./biobert_embeddings"
os.makedirs(output_dir, exist_ok=True)

saved_count = 0  # Count of successfully saved embeddings

# Read drug descriptions and create embeddings
with open('drugbank_descriptions.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    for row in reader:
        drug_id = row['DrugBank ID']
        drug_name = row['Name'] if row['Name'] else "Unknown"
        drug_description = row['Description'] if row['Description'] else ""

        if not drug_description.strip():
            print(f"{drug_id} description doesn't exist.")
            continue

        # Combine name and description
        if drug_name not in drug_description:
            combined_text = f"DrugName: {drug_name}. Description: {drug_description}"
        else:
            combined_text = drug_description

        # Skip long texts with > 1 chunk
        chunks = chunk_text(combined_text)
        if len(chunks) > 1:
            print(f"{drug_id} skipped: too long (chunk count = {len(chunks)})")
            continue

        try:
            embedding = get_biobert_embeddings(combined_text)
            save_path = os.path.join(output_dir, f"{drug_id}.pt")
            torch.save(embedding, save_path)
            saved_count += 1
            print(f"{drug_id} embedding saved:", save_path)
        except Exception as e:
            print(f"Error processing {drug_id}: {e}")
            continue

print("\n Making BioBERT embeddings completed.")
print(f"Total embeddings saved: {saved_count}")
