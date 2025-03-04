import os 
import re

def write_prediction_into_file(model_answer, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(model_answer+'\n')

def get_text_after_word(text, word):
    index = text.find(word)
    if index != -1:
        return text[index + len(word):].strip()
    else:
        return None 
    
def get_text_before_word(text, word):
    index = text.find(word)
    if index != -1:
        return text[:index]
    else:
        return None
    
def get_node_id_from_from_text(model_answer):
    answer = get_text_before_word(model_answer, r'###Input:')
    numbers = re.findall(r'\d+', answer)
    return numbers

def get_response_from_text(model_answer):
    response = get_text_after_word(model_answer, r'###Response:')
    response = get_text_before_word(response, r'###Input:')
    return response

def write_nodes_id_into_file(nodes_list, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        for node_id in nodes_list:
            file.write(str(node_id)+'\t')
        file.write('\n')
