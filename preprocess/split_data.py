import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split

# Combine all the csv files
all_files = glob.glob("paths_length_*.csv")

positive_data = []
negative_data = []

for filename in all_files:
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        if "The answer is yes." in row['output_text']:
            positive_data.append(row)
        else:
            negative_data.append(row)

min_length = min(len(positive_data), len(negative_data))
positive_data = positive_data[:min_length]
negative_data = negative_data[:min_length]

combined_data = positive_data + negative_data

combined_data = pd.DataFrame(combined_data).sample(frac=1, random_state=42).reset_index(drop=True)

positive_instances = combined_data[combined_data['output_text'].str.contains("The answer is yes.")].reset_index(drop=True)
negative_instances = combined_data[combined_data['output_text'].str.contains("The answer is no.")].reset_index(drop=True)

if len(positive_instances) < 2 or len(negative_instances) < 2:
    raise ValueError("Not enough instances to perform stratified split.")

train_pos, test_pos = train_test_split(positive_instances, test_size=0.2, random_state=42)
train_neg, test_neg = train_test_split(negative_instances, test_size=0.2, random_state=42)

train_data = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
test_data = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

train_pos, val_pos = train_test_split(train_pos, test_size=0.2, random_state=42)
train_neg, val_neg = train_test_split(train_neg, test_size=0.2, random_state=42)

val_data = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

train_data.to_csv("train_data.csv", index=False)
val_data.to_csv("val_data.csv", index=False)

test_data['num_nodes'] = test_data['input_text'].apply(lambda x: x.count('node_') // 2 + 1)

for num_nodes in range(2, 7):
    test_subset = test_data[test_data['num_nodes'] == num_nodes].copy()
    if not test_subset.empty:
        if num_nodes == 2:
            test_subset['Prompt'] = test_subset['Prompt'].apply(lambda x: x.split("###Input:")[0] + "###Response:")
        else:
            test_subset['Prompt'] = test_subset['Prompt'].apply(lambda x: x.split("###Response:")[0] + "###Response:")
        test_subset.to_csv(f"test_{num_nodes}_nodes.csv", index=False)

print(f"Training data: {len(train_data)} instances")
print(f"Validation data: {len(val_data)} instances")
for num_nodes in range(2, 7):
    print(f"Test data with {num_nodes} nodes: {len(test_data[test_data['num_nodes'] == num_nodes])} instances")
