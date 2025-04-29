import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score
import torch
import os
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
import utils

# ! Clear gpu storage
torch.cuda.empty_cache()

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

def load_model(model_name, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

y_true = []
y_pred = []

model_name = "meta-llama/Llama-2-7b-hf"
bnb_config = create_bnb_config()
model, tokenizer = load_model(model_name, bnb_config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_merged_dir = "./results/llama/final_merged_checkpoint"
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.save_pretrained(output_merged_dir)

# Loading csv files
csv_file = './content/augmented_test.csv'
df = pd.read_csv(csv_file)

# test limitation
test_limit = 200
accurate_count = 0
if not os.path.exists('./prediction'):
    os.mkdir('./prediction')
else:
    for filename in os.listdir('./prediction'):
        file_path = os.path.join('./prediction', filename)
        try:
            # check path is file or link
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # delete files or links
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
prediction_nodes = []
pridiction_result = []
for index, row in df.iterrows():
    if index >= test_limit:  
          break
    input_text = row['input_text']

    # model answer question
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"], max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
    model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    nodes_id = utils.get_node_id_from_from_text(model_answer)
    response = utils.get_response_from_text(model_answer)
    model_has_yes = 'the answer is yes' in response.lower()
    if model_has_yes:
        utils.write_prediction_into_file(response, './prediction/prediction_yes.txt')
        prediction_nodes.append(nodes_id)
        pridiction_result.append(1)
    else:
        utils.write_prediction_into_file(response, './prediction/prediction_no.txt')
        prediction_nodes.append(nodes_id)
        pridiction_result.append(0)
    
    expected_has_yes = 'yes' in row['output_text'].lower()

    y_true.append(expected_has_yes)
    y_pred.append(model_has_yes)

    if model_has_yes == expected_has_yes:
        accurate_count += 1
        print(accurate_count)

    print(index)

# calculate accuracy
f1 = f1_score(y_true, y_pred, pos_label=True)
print(y_pred)
print(y_true)
accuracy = accurate_count / min(len(df), test_limit)
print(f'Accuracy: {accuracy:.3f}')
print(f'F1 Score: {f1:.3f}')

node_id2hash_data = np.load('./WF/node_id.npy', allow_pickle=True).item()
node_hash2name_data = np.load('./WF/hash_node.npy', allow_pickle=True).item()

with open('./prediction/prediction_result.txt', 'a', encoding='utf-8') as file:
    index = 0
    for index in range(0, len(y_pred)):
        new_line = ''
        nodes = prediction_nodes[index]
        for node in nodes:
            hash_id = node_id2hash_data[int(node)]
            node_name = node_hash2name_data[hash_id]
            new_line = new_line + node_name + '\t'
        if pridiction_result[index] == 1:
            new_line = new_line + 'has connection\t'
        else:
            new_line = new_line + 'has no connection\t'
        if y_pred[index] == y_true[index]:
            new_line = new_line + 'correct\n'
        else:
            new_line = new_line + 'not correct\n'
        file.write(new_line)

# with open('./prediction/node_yes.txt', 'a', encoding='utf-8') as file:
#     for yes_node in yes_nodes:
#         new_line = ''
#         for node in yes_node:
#             hash_id = node_id2hash_data[int(node)]
#             node_name = node_hash2name_data[hash_id]
#             new_line = new_line + node_name + '\t'
#         file.write(new_line+'\n')

# with open('./prediction/node_no.txt', 'a', encoding='utf-8') as file:
#     for no_node in no_nodes:
#         new_line = ''
#         for node in no_node:
#             hash_id = node_id2hash_data[int(node)]
#             node_name = node_hash2name_data[hash_id]
#             new_line = new_line + node_name + '\t'
#         file.write(new_line+'\n')
