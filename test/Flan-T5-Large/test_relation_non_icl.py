import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from transformers import T5ForConditionalGeneration, T5Tokenizer


model_path = '/content/results/checkpoint-3000'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

csv_file = '/content/relation.csv'
df = pd.read_csv(csv_file)

# 初始化计数器和结果列表
tp = fp = fn = 0
y_true = []
y_pred = []
test_limit = 500

for index, row in df.iterrows():
    if index <= test_limit:  # 检查是否已达到测试数量的限制
        break
    input_text = row['input_text']
    expected_word = row['output_text'].rstrip('.').split()[-1].lower()

    # 使用模型生成回答
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

    # 检查模型回答中是否包含最后一个单词
    contains_word = expected_word in model_answer.rstrip('.').split()

    # 更新计数器
    if contains_word:
        tp += 1
        y_pred.append(1)
    else:
        fn += 1
        y_pred.append(0)
    y_true.append(1)

# 计算准确度和F1分数
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f'Accuracy: {accuracy:.3f}')
print(f'F1 Score: {f1:.3f}')
