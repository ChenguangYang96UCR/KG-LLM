import pandas as pd
import json
from collections import deque
import csv
import os

input_file = open("./WF/test/test2id.txt", "r")

# total number of lines
number = int(input_file.readline())

nodes = set()
graph = {}

for i in range(number):
    content = input_file.readline()
    node1, node2, relation = content.strip().split()
    nodes.add(node1)
    relation = int(relation)

    if node1 not in graph:
        graph[node1] = {}
    graph[node1][node2] = relation

node_list = list(nodes)
relation2id = {}

with open("./WF/relation2id.txt", "r") as file:
    relations = int(file.readline())
    for line in file:
        relation, relation_id = line.strip().split("\t")
        relation2id[int(relation_id)] = relation

entity2id = {}
with open("./WF/test/entity2id.txt", "r") as file:
    file.readline()  # Skip the first line (number of entities)
    for line in file:
        entity_name, entity_id = line.strip().split("\t")
        entity2id[int(entity_id)] = entity_name

fieldnames = ['Prompt', 'input_text', 'output_text']
instruction = 'Answer the following yes/no question by reasoning step-by-step. Answer the question by reasoning step-by-step. Choose from the given options:\n1. Yes\n2. No'

def save_to_csv(filename, data):
    with open(filename, mode="w", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def generate_all_paths(graph, max_length):
    def dfs(current_node, path, length):
        if length > max_length:
            return
        if 2 <= length <= max_length:
            paths[length].append(path.copy())
        for neighbor in graph.get(current_node, {}):
            if neighbor not in path:
                path.append(neighbor)
                dfs(neighbor, path, length + 1)
                path.pop()

    paths = {i: [] for i in range(2, max_length + 1)}
    for node in graph:
        dfs(node, [node], 1)
    return paths

def convert_path_to_natural_language(path):
    input_text = ""
    output_text = ""
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
        relation = graph[node1][node2]
        r = relation2id[relation]
        node1_entity = entity2id[int(node1)]
        node2_entity = entity2id[int(node2)]
        input_text += f'node_{node1} has relation_{relation} with node_{node2}. '
        # output_text += f'node_{node1} has relation_{relation} with node_{node2}, means node_{node1} {r} node_{node2}. '
        output_text += f'node_{node1} has relation_{relation} with node_{node2}, means entity_{node1_entity} {r} entity_{node2_entity}. '

    return input_text, output_text

def process_paths(paths):
    for length, path_list in paths.items():
        data = []
        for path in path_list:
            input_text, output_text = convert_path_to_natural_language(path)
            first_node = path[0]
            last_node = path[-1]
            prompt = f'Is node {first_node} connected with node {last_node}?'

            if last_node in graph.get(first_node, {}):
                ground_truth_answer = "The answer is yes."
            else:
                ground_truth_answer = "The answer is no."

            final_output_text = f"###Response:\n{output_text} {ground_truth_answer}"
            comb = f"###Instruction:\nBelow is the detail of a knowledge graph path.\n{prompt}\n{instruction}\n\n###Input:\n{input_text}\n\n###Response:"
            data.append({'Prompt': comb, 'input_text': input_text, 'output_text': final_output_text})
        save_to_csv(f"paths_length_{length}.csv", data)

max_path_length = 6
paths = generate_all_paths(graph, max_path_length)
process_paths(paths)
