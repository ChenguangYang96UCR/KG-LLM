import os 

def filter_entity_data(file_path):
    with open(file_path, 'r+', encoding='utf-8') as file:
        index = 0
        lines = file.readlines()
        for line in lines:
            extract_Line = line.rstrip()
            data = extract_Line.split('\t')
            if len(data) >=2 :
                print(int(data[1]))
                if int(data[1]) >= 1000:
                    print('true')
                    lines.remove(line)
            else:
                # del lines[index]
                lines.remove(line)
            
            # index = index + 1
    
    with open(file_path, 'w+', encoding='utf-8') as input:
        input.writelines(lines)

def filter_test_data(file_path):
    with open(file_path, 'r+', encoding='utf-8') as file:
        index = 0
        lines = file.readlines()
        for line in lines:
            extract_Line = line.rstrip()
            data = extract_Line.split(' ')
            if len(data) >=3 :
                # print(int(data[1]))
                if int(data[0]) >= 1000 or int(data[1]) >= 1000:
                    print('true')
                    lines.remove(line)
            else:
                # del lines[index]
                lines.remove(line)
            
            # index = index + 1
    
    with open(file_path, 'w+', encoding='utf-8') as input:
        input.writelines(lines)


def filter_train_data(file_path):
    with open(file_path, 'r+', encoding='utf-8') as file:
        index = 0
        lines = file.readlines()
        for line in lines:
            extract_Line = line.rstrip()
            data = extract_Line.split(' ')
            if len(data) >=3 :
                # print(int(data[1]))
                if int(data[0]) >= 1000 or int(data[1]) >= 1000:
                    print('true')
                    lines.remove(line)
            else:
                # del lines[index]
                lines.remove(line)
            
            # index = index + 1
    
    with open(file_path, 'w+', encoding='utf-8') as input:
        input.writelines(lines)

if __name__ == '__main__':
    # filter_entity_data('/home/cyang314/KG-LLM-FED0/WN18RR_copy/entity2id.txt')
    # filter_test_data('/home/cyang314/KG-LLM-FED0/WN18RR_copy/test2id.txt')
    filter_train_data('/home/cyang314/KG-LLM-FED0/WN18RR_copy/train2id.txt')
    