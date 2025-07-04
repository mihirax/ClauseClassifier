import os
import gradio as gr
import pdfplumber
import docx
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ Load multilabel model
model_path = "smihira/multi-clause-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
device = 0 if torch.cuda.is_available() else -1
model.eval()

CLASS_NAMES = [
    "anti-assignment",
    "audit rights",
    "cap on liability",
    "governing law",
    "insurance",
    "license grant",
    "minimum commitment",
    "post-termination services",
    "revenue/profit sharing",
    "termination for convenience"
]

THRESHOLD = 0.1

# Global memory
latest_df = pd.DataFrame()
last_embeddings = []

# 🧾 Extract text
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    return ""

# 🔍 Predict
def classify_multilabel(uploaded_file):
    global latest_df, last_embeddings, last_uploaded_name

    text = extract_text(uploaded_file.name)
    last_uploaded_name = os.path.splitext(os.path.basename(uploaded_file.name))[0] 

    text = extract_text(uploaded_file.name)
    clauses = [line.strip() for line in text.split("\n") if len(line.strip()) > 10]

    records = []
    embeddings = []

    for clause in clauses:
        try:
            inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            labels = [label for label, p in zip(CLASS_NAMES, probs) if p >= THRESHOLD]
            confs = [round(p, 3) for p in probs if p >= THRESHOLD]

            label_str = ", ".join(labels) if labels else "None"
            conf_str = ", ".join([f"{c:.2f}" for c in confs]) if confs else "0.00"

            records.append([clause, label_str, conf_str])
            embeddings.append(embedder.encode(clause, convert_to_tensor=True))

        except Exception as e:
            records.append([clause, "ERROR", "0.0"])
            embeddings.append(embedder.encode("ERROR", convert_to_tensor=True))

    latest_df = pd.DataFrame(records, columns=["Clause", "Predicted Labels", "Confidences"])
    last_embeddings = embeddings

    return latest_df

# 💾 CSV Download
def download_csv():
    global latest_df, last_uploaded_name
    filename = f"{last_uploaded_name}.csv"
    latest_df.to_csv(filename, index=False)
    return filename

# 📈 Explain prediction
def explain_multilabel(index):
    clause = latest_df.iloc[int(index)]["Clause"]
    inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        probs = torch.sigmoid(model(**inputs).logits).squeeze().cpu().numpy()
    return pd.DataFrame({
        "Label": CLASS_NAMES,
        "Confidence": [round(p, 3) for p in probs]
    }).sort_values("Confidence", ascending=False)

# 🔁 Similarity
def find_similar_clauses(index):
    query_emb = last_embeddings[int(index)]
    sims = util.pytorch_cos_sim(query_emb, torch.stack(last_embeddings))[0]
    top5 = torch.topk(sims, k=5)
    rows = []
    for i in top5.indices.tolist():
        rows.append({
            "Clause": latest_df.iloc[i]["Clause"],
            "Predicted Labels": latest_df.iloc[i]["Predicted Labels"],
            "Similarity": round(float(sims[i]), 3)
        })
    return pd.DataFrame(rows)

# 🎛 Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 Multilabel Contract Clause Classifier")

    file_input = gr.File(label="📄 Upload Contract (.pdf or .docx)")
    output_table = gr.Dataframe(headers=["Clause", "Predicted Labels", "Confidences"], interactive=False)
    download_btn = gr.Button("📥 Download CSV")
    file_output = gr.File(label="📁 Saved Output File")

    with gr.Row():
        clause_index = gr.Number(label="🔢 Clause Index", value=0)
        explain_btn = gr.Button("📈 Explain Prediction")
        similar_btn = gr.Button("🔁 Similar Clauses")

    explain_out = gr.Dataframe(headers=["Label", "Confidence"], interactive=False)
    similar_out = gr.Dataframe(headers=["Clause", "Predicted Labels", "Similarity"], interactive=False)

    file_input.change(classify_multilabel, inputs=file_input, outputs=output_table)
    download_btn.click(download_csv, outputs=file_output)
    explain_btn.click(explain_multilabel, inputs=clause_index, outputs=explain_out)
    similar_btn.click(find_similar_clauses, inputs=clause_index, outputs=similar_out)

demo.launch()
