import os
import gradio as gr
import pdfplumber
import docx
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# 🔹 Load classification model & tokenizer
model = AutoModelForSequenceClassification.from_pretrained("models/clause_classifier")
tokenizer = AutoTokenizer.from_pretrained("models/clause_classifier")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# 🔹 Load Sentence-BERT for similarity
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Label map
LABEL_MAP = {
    "LABEL_0": "anti-assignment",
    "LABEL_1": "audit rights",
    "LABEL_2": "cap on liability",
    "LABEL_3": "exclusivity",
    "LABEL_4": "governing law",
    "LABEL_5": "insurance",
    "LABEL_6": "license grant",
    "LABEL_7": "minimum commitment",
    "LABEL_8": "post-termination services",
    "LABEL_9": "revenue/profit sharing"
}

# Global variable to hold last dataframe
latest_df = pd.DataFrame()
last_embeddings = None

# 📥 Extract contract text
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    return ""

# 🔍 Classify and store
def classify_uploaded_contract(uploaded_file):
    global latest_df, last_embeddings, last_uploaded_name

    text = extract_text(uploaded_file.name)
    last_uploaded_name = os.path.splitext(os.path.basename(uploaded_file.name))[0] 

    text = extract_text(uploaded_file.name)
    clauses = [line.strip() for line in text.split("\n") if len(line.strip()) > 10]

    data = []
    embeddings = []
    for clause in clauses:
        try:
            preds = classifier(clause)[0]
            top = max(preds, key=lambda x: x["score"])
            label = LABEL_MAP.get(top["label"], top["label"])
            data.append([clause, label, round(top["score"], 3)])
            embeddings.append(embedder.encode(clause, convert_to_tensor=True))
        except:
            data.append([clause, "ERROR", 0.0])
            embeddings.append(embedder.encode("ERROR", convert_to_tensor=True))

    latest_df = pd.DataFrame(data, columns=["Clause", "Predicted Label", "Confidence"])
    last_embeddings = embeddings
    return latest_df

# 📤 Download
def download_csv():
    global latest_df, last_uploaded_name
    filename = f"{last_uploaded_name}.csv"
    latest_df.to_csv(filename, index=False)
    return filename

# 📊 Explain a clause
def explain_clause(index):
    global latest_df
    clause = latest_df.iloc[index]["Clause"]
    predictions = classifier(clause)[0]
    return pd.DataFrame([
        {"Label": LABEL_MAP.get(p["label"], p["label"]), "Confidence": round(p["score"], 3)}
        for p in sorted(predictions, key=lambda x: x["score"], reverse=True)
    ])

# 🔁 Find similar clauses
def find_similar_clauses(index):
    global latest_df, last_embeddings
    if last_embeddings is None: return pd.DataFrame()
    query_emb = last_embeddings[index]
    sims = util.pytorch_cos_sim(query_emb, torch.stack(last_embeddings))[0]
    top5 = torch.topk(sims, k=5)
    rows = []
    for i in top5.indices.tolist():
        rows.append({
            "Clause": latest_df.iloc[i]["Clause"],
            "Label": latest_df.iloc[i]["Predicted Label"],
            "Similarity": round(float(sims[i]), 3)
        })
    return pd.DataFrame(rows)

# 🎛 Gradio App
with gr.Blocks() as demo:
    gr.Markdown("# Single Contract Clause Classifier")

    file_input = gr.File(label="📄 Upload Contract (.pdf or .docx)")
    output_table = gr.Dataframe(headers=["Clause", "Predicted Label", "Confidence"], interactive=False)
    download_btn = gr.Button("📥 Download CSV")
    file_output = gr.File(label="Download File")

    with gr.Row():
        clause_index = gr.Number(label="🔢 Clause Row Index", value=0)
        explain_btn = gr.Button("📈 Explain Prediction")
        similar_btn = gr.Button("🔁 Show Similar Clauses")

    explain_output = gr.Dataframe(headers=["Label", "Confidence"], interactive=False)
    similar_output = gr.Dataframe(headers=["Clause", "Label", "Similarity"], interactive=False)

    # 🔁 Function bindings
    file_input.change(classify_uploaded_contract, inputs=file_input, outputs=output_table)
    download_btn.click(download_csv, outputs=file_output)
    explain_btn.click(explain_clause, inputs=clause_index, outputs=explain_output)
    similar_btn.click(find_similar_clauses, inputs=clause_index, outputs=similar_output)

demo.launch()
