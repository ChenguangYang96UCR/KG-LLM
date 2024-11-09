from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model_id = "meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

llama2_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
)

def add_special_tokens(tokenizer, entity_file, relation_file):
    with open(entity_file, "r") as ef:
        ef.readline()
        node_ids = [f"node_{line.strip().split()[0]}" for line in ef]

    with open(relation_file, "r") as rf:
        rf.readline()
        relation_ids = [f"relation_{line.strip().split()[1]}" for line in rf]

    special_tokens = node_ids + relation_ids
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    return tokenizer

entity_file = "/content/entity2id.txt"
relation_file = "/content/relation2id.txt"
llama2_tokenizer = add_special_tokens(llama2_tokenizer, entity_file, relation_file)

base_model.resize_token_embeddings(len(llama2_tokenizer))

llama2_model = PeftModel.from_pretrained(base_model, "llama2_checkpoint")

def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(model, tokenizer, test_file):
    df = pd.read_csv(test_file)
    prompts = df['Prompt'].tolist()
    ground_truths = df['output_text'].apply(lambda x: 'yes' in x).astype(int).tolist()
    print(ground_truths)

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
    f1, auc = evaluate_model(llama2_model, llama2_tokenizer, test_file)
    print(f"Results for {test_file}:")
    print(f"  F1 Score: {f1}")
    print(f"  AUC Score: {auc}")