import pandas as pd
import json
from collections import deque
import csv
import random
from sklearn.model_selection import train_test_split
import os
import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

input_file = open("train2id.txt", "r")
output_file = open("output.txt", "w")

# total number of lines
number = int(input_file.readline())

nodes = set()

graph = {}

for i in range(number):
    content = input_file.readline()
    node1, node2, relation = content.strip().split()

    nodes.add(node1)
    # nodes.add(node2)

    relation = int(relation)

    # Check if the first node already exists in the dictionary
    if node1 not in graph:
        # If not, create a new dictionary for the node
        graph[node1] = {}
    # Add the neighboring node and the relationship to the dictionary for node1
    graph[node1][node2] = relation


node_list = list(nodes)
node_list2 = list(nodes)



# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# relation2id = {}

# with open("relation2id.txt", "r") as file:
#     relations = int(file.readline())
#     for line in file:
#         relation, relation_id = line.strip().split("\t")
#         relation2id[int(relation_id)] = relation

unique_rows = set()

# change size if you want, but larger the size, smaller the data
size = 30
# how many positive and negative data set you want to train, the more dataset, the more time to train
total = 60000
fieldnames = ['Prompt', 'input_text', 'output_text', 'train_input_text', 'train_output_text']

# instruction = 'Answer the following yes/no question by reasoning step-by-step. '

with open("train_data.csv", mode="w", newline='') as tra:
  with open("positive_data.csv", mode="w", newline='') as pos:
    with open("negative_data.csv",  mode="w", newline='') as neg:

        # Create a CSV writer object and write the headers to the file
        writer_pos = csv.DictWriter(pos, fieldnames=fieldnames)
        writer_pos.writeheader()

        writer_neg = csv.DictWriter(neg, fieldnames=fieldnames)
        writer_neg.writeheader()

        writer_tra = csv.DictWriter(tra, fieldnames=fieldnames)
        writer_tra.writeheader()


        def dfs(graph, size):
            pos_count = 0
            neg_count = 0
            times = 0
            term = True

            while times < total:
                visited = set()
                kg = []
                graph_size = random.randint(2, size)
                first_node = random.choice(node_list)
                visited.add(first_node)
                last_node = ""
                previous_node = first_node
                stack = [first_node]
                input_text = ""
                output_text = ""
                while len(visited) < graph_size:
                    if previous_node not in graph or set(graph[previous_node].keys()).issubset(visited):
                        node = random.choice(node_list)
                        while node in visited:
                            node = random.choice(node_list)
                        input_text += 'node_{} not connected with node_{}. '.format(previous_node, node)
                        # output_text += 'node_{} not connected with node_{} means there is no relationship between node_{} and node_{}. '.format(previous_node, node, previous_node, node)
                        visited.add(node)
                        previous_node = node
                    else:
                        node = random.choice(list(graph[previous_node].keys()))
                        while node in visited:
                            node = random.choice(list(graph[previous_node].keys()))
                        relation = graph[previous_node][node]
                        # r = relation2id[relation]
                        input_text += 'node_{} has relation_{} with node_{}. '.format(previous_node, relation, node)
                        # output_text += 'node_{} has relation_{} with node_{}, means node_{} {} node_{}. '.format(previous_node, relation, node, previous_node, r, node)
                        visited.add(node)
                        previous_node = node
                    if len(visited) == graph_size:
                        last_node = previous_node

                # input_text += 'Answer the following yes/no question by reasoning step-by-step. Is the first node connnected with the last node?'
                was = len(unique_rows)
                unique_rows.add(input_text)
                if len(unique_rows) > was:
                  # if pos_count < int(total/2):
                    if first_node in graph and last_node in graph[first_node] and term:
                      # if pos_count < int(total/2):
                        # final_relation = relation2id[graph[first_node][last_node]]
                        output_text += 'The answer is yes.'
                        prompt = 'Is node {} connnected with node {}?'.format(first_node, last_node)
                        comb = "###Input:\n" + input_text + prompt + "\n\n###Response:\n" + output_text
                        writer_pos.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text' : "###Input:\n" + input_text + prompt, 'train_output_text' : "###Response:\n" + output_text})
                        writer_tra.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text' : "###Input:\n" + input_text + prompt, 'train_output_text' : "###Response:\n" + output_text})
                        pos_count += 1
                        term = False
                        times += 1
                    elif last_node in graph and first_node in graph[last_node] and term:
                      # if pos_count < int(total/2):
                        # final_relation = relation2id[graph[last_node][first_node]]
                        output_text += 'The answer is yes.'
                        prompt = 'Is node {} connected with node {}?'.format(last_node, first_node)
                        comb = "###Input:\n" + input_text + prompt + "\n\n###Response:\n" + output_text
                        writer_pos.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text' : "###Input:\n" + input_text + prompt, 'train_output_text' : "###Response:\n" + output_text})
                        writer_tra.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text' : "###Input:\n" + input_text + prompt, 'train_output_text' : "###Response:\n" + output_text})
                        pos_count += 1
                        term = False
                        times += 1
                  # elif neg_count < int(total//2):
                    elif not term:
                      # if neg_count < int(total/2):
                        output_text += 'The answer is no.'
                        prompt = 'Is node {} connected with node {}?'.format(first_node, last_node)
                        comb = "###Input:\n" + input_text + prompt + "\n\n###Response:\n" + output_text
                        writer_neg.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text' : "###Input:\n" + input_text + prompt, 'train_output_text' : "###Response:\n" + output_text})
                        writer_tra.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text' : "###Input:\n" + input_text + prompt, 'train_output_text' : "###Response:\n" + output_text})
                        neg_count += 1
                        term = True
                        times += 1
                else:
                  continue


            print(pos_count)
            print(neg_count)

        dfs(graph, size)




# 读取CSV文件
positive_df = pd.read_csv('positive_data.csv')
negative_df = pd.read_csv('negative_data.csv')

# 各自分割数据集为80%的训练集和20%的测试集
train_pos, test_pos = train_test_split(positive_df, test_size=0.2, random_state=42)
train_neg, test_neg = train_test_split(negative_df, test_size=0.2, random_state=42)

# 合并训练集和测试集
train_df = pd.concat([train_pos, train_neg])
test_df = pd.concat([test_pos, test_neg])

# 打乱训练集和测试集
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

import pandas as pd

# 加载CSV文件
csv_file = 'test.csv'  # 请替换为你的CSV文件名
df = pd.read_csv(csv_file)

# 在所有的'input_text'数据前面加上"###Input:"
df['input_text'] = "###Input: \n" + df['input_text']

# 保存更改后的DataFrame到新的CSV文件
df.to_csv('modified_' + csv_file, index=False)

print("Updated CSV file has been saved as 'modified_" + csv_file + "'.")


df = pd.read_csv('test.csv')
count = len(df)

c = 0

fieldnames = ['input_text', 'output_text']
with open("train.csv") as train:
  with open("context.csv", mode="w", newline='') as context:
      writer_context = csv.DictWriter(context, fieldnames=fieldnames)
      writer_context.writeheader()
      reader_train = csv.reader(train)
      # Iterate over each row in the CSV file
      for row in reader_train:
          input = row[1]
          output = row[2]
          if input == 'input_text' or output == 'output_text':
            continue

          input = input.split('.')
          input = input[:-1]
          input = '.'.join(input)

          output = output.split('.')
          words = output[-3:-1]
          # relationship = words[3]
          # print(words)
          words = '.'.join(words)

          if c < count and len(input) < 30:
            cont = '###Context: {}.\n\n###Input:'.format(input + '.' + words)
            writer_context.writerow({'input_text': cont, 'output_text': words})
            c += 1
            # print(c)




# 加载CSV文件
context_df = pd.read_csv('context.csv')
test_df = pd.read_csv('test.csv')

# 对test.csv中的每个input_text随机添加一个context.csv中的input_text
for index, row in test_df.iterrows():
    # 随机选择一个context
    random_context = random.choice(context_df['input_text'])
    # 将context和test中的文本合并，并直接覆盖原有的input_text
    test_df.at[index, 'input_text'] = random_context + " " + row['input_text']

# 保存更新后的test_df到CSV文件
test_df.to_csv('augmented_test.csv', index=False)

print("Augmented test CSV file has been created.")

import csv
import pandas as pd
# context = 'Given node_2075 has relation_4 with node_12648. The relation between the first node and last node is _member_meronym. '
# instruction = 'Answer the following multiple-choice question by choosing one of these options: '
# option = ''
instruction = 'Answer the following question step-by-step. '
count = 0

question = 'The relationship between the first node and the last node is ?'
fieldnames = ['input_text', 'output_text']
with open("relation2id.txt", "r") as file:
    relations = int(file.readline())
    for line in file:
        relation, relation_id = line.strip().split("\t")
    #     instruction += relation + ', '
    # instruction = instruction[:-2] + '. '

with open("test.csv") as test:
    with open("relation.csv", mode="w", newline='') as icl:
      writer_icl = csv.DictWriter(icl, fieldnames=fieldnames)
      writer_icl.writeheader()
      reader = csv.reader(test)


      for row in reader:
          # print(row)
          input = row[1]
          output = row[2]
          if input == 'input_text' or output == 'output_text':
            continue
          input = input.split('.')
          input = input[:-1]
          input = '.'.join(input)
          # print(input)
          input = instruction + input + '. ' + question
          # input = input + '. ' + instruction + question

          output = output.split('.')
          words = output[-3].split()
          relationship = words[3]
          # print(relationship)
          output = output[:-2]
          output = '.'.join(output)
          # print(output)
          output += '. The relationship between the first node and the last node is {}.'.format(relationship)
          # print(output)
          if len(input) < 550:
            writer_icl.writerow({'input_text': input, 'output_text': output})
            count+=1

import csv
import pandas as pd
# context = 'Given node_2075 has relation_4 with node_12648. The relation between the first node and last node is _member_meronym. '
# instruction = 'Answer the following multiple-choice question by choosing one of these options: '
# option = ''
instruction = 'Answer the following question step-by-step. '
count = 0

question = 'The relationship between the first node and the last node is ?'
fieldnames = ['input_text', 'output_text']
with open("relation2id.txt", "r") as file:
    relations = int(file.readline())
    for line in file:
        relation, relation_id = line.strip().split("\t")
    #     instruction += relation + ', '
    # instruction = instruction[:-2] + '. '

with open("test.csv") as test:
    with open("test_icl.csv", mode="w", newline='') as icl:
      writer_icl = csv.DictWriter(icl, fieldnames=fieldnames)
      writer_icl.writeheader()
      reader = csv.reader(test)


      for row in reader:
          # print(row)
          input = row[1]
          output = row[2]
          if input == 'input_text' or output == 'output_text':
            continue
          input = input.split('.')
          input = input[:-1]
          input = '.'.join(input)
          # print(input)
          input = instruction + input + '. ' + question
          # input = input + '. ' + instruction + question

          output = output.split('.')
          words = output[-3].split()
          relationship = words[3]
          # print(relationship)
          output = output[:-2]
          output = '.'.join(output)
          # print(output)
          output += '. The relationship between the first node and the last node is {}.'.format(relationship)
          # print(output)
          if len(input) < 550:
            writer_icl.writerow({'input_text': input, 'output_text': output})
            count+=1

c = 0

with open("train.csv") as train:
  with open("context.csv", mode="w", newline='') as context:
      writer_context = csv.DictWriter(context, fieldnames=fieldnames)
      writer_context.writeheader()
      reader_train = csv.reader(train)
      # Iterate over each row in the CSV file
      for row in reader_train:
          input = row[1]
          output = row[2]
          if input == 'input_text' or output == 'output_text':
            continue

          input = input.split('.')
          input = input[:-1]
          input = '.'.join(input)

          output = output.split('.')
          words = output[-3].split()
          relationship = words[3]

          if c < count and len(input) < 50:
            cont = '###Context: {}. The relationship between the first node and last node is {}.\n\n###Input:'.format(input, relationship)
            writer_context.writerow({'input_text': cont, 'output_text': relationship})
            c += 1
            # print(c)



# 加载CSV文件
context_df = pd.read_csv('context.csv')
test_df = pd.read_csv('test_icl.csv')

# 对test.csv中的每个input_text随机添加一个context.csv中的input_text
for index, row in test_df.iterrows():
    # 随机选择一个context
    random_context = random.choice(context_df['input_text'])
    # 将context和test中的文本合并，并直接覆盖原有的input_text
    test_df.at[index, 'input_text'] = random_context + " " + row['input_text']

# 保存更新后的test_df到CSV文件
test_df.to_csv('icl_relation.csv', index=False)

print("Augmented test CSV file has been created.")

# read csv files
df1 = pd.read_csv('context.csv')
df2 = pd.read_csv('test_icl.csv')

# combine dataframes by interleaving rows

combined_df = pd.concat([df1, df2]).sort_index(kind='merge')

# write to new csv file
combined_df.to_csv('icl_relation.csv', index=False)