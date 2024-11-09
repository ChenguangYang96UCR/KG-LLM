import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from sklearn.metrics import f1_score, roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained('./results').to(device)
flan_t5_tokenizer = AutoTokenizer.from_pretrained('./results')

def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(device)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(model, tokenizer, test_file):
    df = pd.read_csv(test_file)
    prompts = df['Prompt'].tolist()
    ground_truths = df['output_text'].apply(lambda x: 'yes' in x).astype(int).tolist()

    predictions = []
    for prompt in prompts:
        generated_text = generate_text(model, tokenizer, prompt)
        prediction = 1 if 'yes' in generated_text.lower() else 0
        predictions.append(prediction)

    f1 = f1_score(ground_truths, predictions)
    auc = roc_auc_score(ground_truths, predictions)
    return f1, auc

test_files = ["test_2_nodes.csv", "test_3_nodes.csv", "test_4_nodes.csv", "test_5_nodes.csv", "test_6_nodes.csv"]

for test_file in test_files:
    f1, auc = evaluate_model(flan_t5_model, flan_t5_tokenizer, test_file)
    print(f"Results for {test_file}:")
    print(f"  F1 Score: {f1}")
    print(f"  AUC Score: {auc}")
