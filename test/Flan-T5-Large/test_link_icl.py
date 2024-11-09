import pandas as pd
from sklearn.metrics import f1_score
from transformers import T5ForConditionalGeneration, T5Tokenizer


model_path = '/content/results/checkpoint-3000'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

y_true = []
y_pred = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 读取CSV文件
csv_file = '/content/augmented_test.csv'
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