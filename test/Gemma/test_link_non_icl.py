import pandas as pd
from sklearn.metrics import f1_score
import os
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


y_true = []
y_pred = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoPeftModelForCausalLM.from_pretrained("final_checkpoint", device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = "results/gemma/final_merged_checkpoint"
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
tokenizer.save_pretrained(output_merged_dir)

# 读取CSV文件
csv_file = '/content/modified_test.csv'
df = pd.read_csv(csv_file)

# 限制测试数量
test_limit = 2000
# 计数器
accurate_count = 0

for index, row in df.iterrows():
    if index <= test_limit:  # 检查是否已达到测试数量的限制
          break
    # 获取输入文本
    input_text = row['input_text']

    # 使用模型生成回答
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"], max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
    model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 检查模型回答和期望输出中是否包含'yes'或'Yes'
    model_has_yes = 'yes' in model_answer.lower()
    expected_has_yes = 'yes' in row['output_text'].lower()

    # print("model_has_yes: ", model_has_yes)
    # print("expected_has_yes: ", expected_has_yes)

    y_true.append(expected_has_yes)
    y_pred.append(model_has_yes)


    # 判断准确性
    if model_has_yes == expected_has_yes:
        accurate_count += 1
        print(accurate_count)

    print(index)

# 计算准确率
f1 = f1_score(y_true, y_pred, pos_label=True)
accuracy = accurate_count / min(len(df), test_limit)
print(f'Accuracy: {accuracy:.3f}')
print(f'F1 Score: {f1:.3f}')