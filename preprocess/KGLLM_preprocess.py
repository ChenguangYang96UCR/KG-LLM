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


#! 1. Store training part 
input_file = open("./WF/train2id.txt", "r")
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
relation2id = {}

with open("./WF/relation2id.txt", "r") as file:
    relations = int(file.readline())
    for line in file:
        relation, relation_id = line.strip().split("\t")
        relation2id[int(relation_id)] = relation

unique_rows = set()

# change size if you want, but larger the size, smaller the data
size = 30
# how many positive and negative data set you want to train, the more dataset, the more time to train
total = 60000
fieldnames = ['Prompt', 'input_text', 'output_text', 'train_input_text', 'train_output_text']

instruction = 'Answer the following yes/no question by reasoning step-by-step. '

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
                        output_text += 'node_{} not connected with node_{} means there is no relationship between node_{} and node_{}. '.format(previous_node, node, previous_node, node)
                        visited.add(node)
                        previous_node = node
                    else:
                        node = random.choice(list(graph[previous_node].keys()))
                        while node in visited:
                            node = random.choice(list(graph[previous_node].keys()))
                        relation = graph[previous_node][node]
                        r = relation2id[relation]
                        input_text += 'node_{} has relation_{} with node_{}. '.format(previous_node, relation, node)
                        output_text += 'node_{} has relation_{} with node_{}, means node_{} {} node_{}. '.format(previous_node, relation, node, previous_node, r, node)
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
                        final_relation = relation2id[graph[first_node][last_node]]
                        output_text += 'So node {} {} node {}. The answer is yes.'.format(first_node, final_relation, last_node)
                        prompt = 'Is node {} connnected with node {}?'.format(first_node, last_node)
                        comb = "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text + "\n\n###Response:\n" + output_text
                        writer_pos.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text': "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text, 'train_output_text' : "###Response:\n" + output_text})
                        writer_tra.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text': "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text, 'train_output_text' : "###Response:\n" + output_text})
                        pos_count += 1
                        term = False
                        times += 1
                    elif last_node in graph and first_node in graph[last_node] and term:
                      # if pos_count < int(total/2):
                        final_relation = relation2id[graph[last_node][first_node]]
                        output_text += 'So node {} {} node {}. The answer is yes.'.format(last_node, final_relation, first_node)
                        prompt = 'Is node {} connected with node {}?'.format(last_node, first_node)
                        comb = "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text + "\n\n###Response:\n" + output_text
                        writer_pos.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text': "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text, 'train_output_text' : "###Response:\n" + output_text})
                        writer_tra.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text': "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text, 'train_output_text' : "###Response:\n" + output_text})
                        pos_count += 1
                        term = False
                        times += 1
                  # elif neg_count < int(total//2):
                    elif not term:
                      # if neg_count < int(total/2):
                        output_text += 'So there is no connection between node {} and node {}. The answer is no.'.format(first_node, last_node)
                        prompt = 'Is node {} connected with node {}?'.format(first_node, last_node)
                        comb = "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text + "\n\n###Response:\n" + output_text
                        writer_neg.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text': "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text, 'train_output_text' : "###Response:\n" + output_text})
                        writer_tra.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text': "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text, 'train_output_text' : "###Response:\n" + output_text})
                        neg_count += 1
                        term = True
                        times += 1
                else:
                  continue

            print(pos_count)
            print(neg_count)
        dfs(graph, size)

positive_df = pd.read_csv('positive_data.csv')
negative_df = pd.read_csv('negative_data.csv')

train_pos, test_pos = train_test_split(positive_df,test_size=0.005, train_size=0.995, random_state=42)
train_neg, test_neg = train_test_split(negative_df,test_size=0.005, train_size=0.995, random_state=42)
train_df = pd.concat([train_pos, train_neg])
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df.to_csv('train.csv', index=False)


#! 2. Store test part 
input_file = open("./WF/test/test2id.txt", "r")
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
relation2id = {}

with open("./WF/relation2id.txt", "r") as file:
    relations = int(file.readline())
    for line in file:
        relation, relation_id = line.strip().split("\t")
        relation2id[int(relation_id)] = relation

unique_rows = set()

# change size if you want, but larger the size, smaller the data
size = 30
# how many positive and negative data set you want to train, the more dataset, the more time to train
total = 60000
fieldnames = ['Prompt', 'input_text', 'output_text', 'train_input_text', 'train_output_text']

instruction = 'Answer the following yes/no question by reasoning step-by-step. '

with open("test_data.csv", mode="w", newline='') as tra:
  with open("test_positive_data.csv", mode="w", newline='') as pos:
    with open("test_negative_data.csv",  mode="w", newline='') as neg:

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
                        output_text += 'node_{} not connected with node_{} means there is no relationship between node_{} and node_{}. '.format(previous_node, node, previous_node, node)
                        visited.add(node)
                        previous_node = node
                    else:
                        node = random.choice(list(graph[previous_node].keys()))
                        while node in visited:
                            node = random.choice(list(graph[previous_node].keys()))
                        relation = graph[previous_node][node]
                        r = relation2id[relation]
                        input_text += 'node_{} has relation_{} with node_{}. '.format(previous_node, relation, node)
                        output_text += 'node_{} has relation_{} with node_{}, means node_{} {} node_{}. '.format(previous_node, relation, node, previous_node, r, node)
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
                        final_relation = relation2id[graph[first_node][last_node]]
                        output_text += 'So node {} {} node {}. The answer is yes.'.format(first_node, final_relation, last_node)
                        prompt = 'Is node {} connnected with node {}?'.format(first_node, last_node)
                        comb = "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text + "\n\n###Response:\n" + output_text
                        writer_pos.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text': "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text, 'train_output_text' : "###Response:\n" + output_text})
                        writer_tra.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text': "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text, 'train_output_text' : "###Response:\n" + output_text})
                        pos_count += 1
                        term = False
                        times += 1
                    elif last_node in graph and first_node in graph[last_node] and term:
                      # if pos_count < int(total/2):
                        final_relation = relation2id[graph[last_node][first_node]]
                        output_text += 'So node {} {} node {}. The answer is yes.'.format(last_node, final_relation, first_node)
                        prompt = 'Is node {} connected with node {}?'.format(last_node, first_node)
                        comb = "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text + "\n\n###Response:\n" + output_text
                        writer_pos.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text': "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text, 'train_output_text' : "###Response:\n" + output_text})
                        writer_tra.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text': "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text, 'train_output_text' : "###Response:\n" + output_text})
                        pos_count += 1
                        term = False
                        times += 1
                  # elif neg_count < int(total//2):
                    elif not term:
                      # if neg_count < int(total/2):
                        output_text += 'So there is no connection between node {} and node {}. The answer is no.'.format(first_node, last_node)
                        prompt = 'Is node {} connected with node {}?'.format(first_node, last_node)
                        comb = "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text + "\n\n###Response:\n" + output_text
                        writer_neg.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text': "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text, 'train_output_text' : "###Response:\n" + output_text})
                        writer_tra.writerow({'Prompt': comb, 'input_text': input_text + prompt, 'output_text': output_text, 'train_input_text': "###Instruction:\n" + instruction + prompt + "\n\n###Input:\n" + input_text, 'train_output_text' : "###Response:\n" + output_text})
                        neg_count += 1
                        term = True
                        times += 1
                else:
                  continue

            print(pos_count)
            print(neg_count)

        dfs(graph, size)

positive_df = pd.read_csv('test_positive_data.csv')
negative_df = pd.read_csv('test_negative_data.csv')
train_pos, test_pos = train_test_split(positive_df,train_size=0.005, test_size=0.995, random_state=42)
train_neg, test_neg = train_test_split(negative_df,train_size=0.005, test_size=0.995, random_state=42)
test_df = pd.concat([test_pos, test_neg])
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df.to_csv('test.csv', index=False)


#! 3. Loading csv files
csv_file = 'test.csv' 
df = pd.read_csv(csv_file)
df['input_text'] = "###Input: \n" + df['input_text']
df.to_csv('modified_' + csv_file, index=False)

print("Updated CSV file has been saved as 'modified_" + csv_file + "'.")

df = pd.read_csv('test.csv')
count = len(df)

c = 0

fieldnames = ['input_text', 'output_text']
with open("train.csv", 'r+', encoding='utf-8') as train:
  with open("context.csv", mode="w", newline='', encoding='utf-8') as context:
      writer_context = csv.DictWriter(context, fieldnames=fieldnames)
      writer_context.writeheader()
      reader_train = csv.reader(train)
      # Iterate over each row in the CSV file
      for row in reader_train:
          input = row[0]
          output = row[1]
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

          # if c < count and len(input) < 30:
          if c < count:
            cont = '###Context: {}.\n\n###Input:'.format(input + '.' + words)
            writer_context.writerow({'input_text': cont, 'output_text': words})
            c += 1
            # print(c)


# Loading cvs files
context_df = pd.read_csv('context.csv')
test_df = pd.read_csv('test.csv')

# random add one input text
for index, row in test_df.iterrows():
    # 随机选择一个context
    filtered_numbers = context_df['input_text'][1:]
    random_context = random.choice(filtered_numbers.tolist())
    # combine context and test，covere origin input text
    test_df.at[index, 'input_text'] = random_context + " " + row['input_text']

# save or update to augmented_test file 
if not os.path.exists('./content'):
    os.mkdir('./content')
else:
    for filename in os.listdir('./content'):
        file_path = os.path.join('./content', filename)
        try:
            # check path is file or link
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # delete files or links
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
test_df.to_csv('./content/augmented_test.csv', index=False)

print("Augmented test CSV file has been created.")



# context = 'Given node_2075 has relation_4 with node_12648. The relation between the first node and last node is _member_meronym. '
# instruction = 'Answer the following multiple-choice question by choosing one of these options: '
# option = ''
instruction = 'Answer the following question step-by-step. '
count = 0

question = 'The relationship between the first node and the last node is ?'
fieldnames = ['input_text', 'output_text']
with open("./WF/relation2id.txt", "r") as file:
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


# context = 'Given node_2075 has relation_4 with node_12648. The relation between the first node and last node is _member_meronym. '
# instruction = 'Answer the following multiple-choice question by choosing one of these options: '
# option = ''
instruction = 'Answer the following question step-by-step. '
count = 0

question = 'The relationship between the first node and the last node is ?'
fieldnames = ['input_text', 'output_text']
with open("./WF/relation2id.txt", "r") as file:
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


# Loading csv files
context_df = pd.read_csv('context.csv')
test_df = pd.read_csv('test_icl.csv')

for index, row in test_df.iterrows():
    random_context = random.choice(context_df['input_text'])
    test_df.at[index, 'input_text'] = random_context + " " + row['input_text']

test_df.to_csv('icl_relation.csv', index=False)

print("Augmented test CSV file has been created.")

# read csv files
df1 = pd.read_csv('context.csv')
df2 = pd.read_csv('test_icl.csv')

# combine dataframes by interleaving rows

combined_df = pd.concat([df1, df2]).sort_index(kind='merge')

# write to new csv file
combined_df.to_csv('./content/icl_relation.csv', index=False)

