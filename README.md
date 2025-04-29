# KG-LLM: Knowledge Graph Large Language Model for Link Prediction

![](KG.png?raw=true)

## Abstract:
The task of multi-hop link prediction within knowledge graphs (KGs) stands as a challenge in the field of knowledge graph analysis, a challenge increasingly resolvable due to advancements in natural language processing (NLP) and KG embedding techniques. This paper introduces a novel methodology, the Knowledge Graph Large Language Model Framework (KG-LLM), which leverages pivotal NLP paradigms. We use our method to convert structured knowledge graph data into natural language, and then use these natural language prompts to fine-tune large language models (LLMs) to enhance multi-hop link prediction in KGs. By converting the KG to natural language prompts, our framework is designed to discern and learn the latent representations of entities and their interrelations. To show the efficacy of the KG-LLM Framework, we fine-tune three leading LLMs within this framework, using both in-context learning (ICL) and without ICL techniques for a thorough evaluation. Further, we explore the framework's potential to provide LLMs with zero-shot capabilities for handling previously unseen prompts. Our experimental findings discover that the performance of our approach can significantly boost the models' generalization capacity, thereby ensuring more precise predictions in unfamiliar scenarios.

## Requirements:
- Set up your Python environment
  
  > Install the requirements.txt file by running
  > ```python
  > pip install -r requirements.txt
  > ```

- Requesting model access from META and Google

  > Visit this [link](https://ai.meta.com/llama/) and request the access to the **Llama-2** models
  
  > Visit this [link](https://blog.google/technology/developers/gemma-open-models/) and request the access to the **Gemma** models

- Requesting model access from HuggingFace

  > Once request is approved, use the same email adrress to get the access of the model from HF [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b) and [Gemma](https://huggingface.co/google/gemma-7b).

- Authorising HF token

  > Once HF request to access the model has been approved, create hugging face token [here](https://huggingface.co/settings/tokens)

  > Run below code and enter your token. It will authenticate your HF account
  ```python
  >>> huggingface-cli login
  
  or
  
  >>> from huggingface_hub import login
  >>> login(YOUR_HF_TOKEN)
  ```

## Dataset:
We conduct experiments over *two* real-world datasets, WN18RR and NELL-995, which are constructed and released by the [OpenKE library](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch).
| Dataset  | #Entities | #Triples  | #Relations  |
| ------------- |:-------------:| -----:| -----:|
| [WN18RR](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/WN18RR)      | 40,943 | 86,835 | 11 |
| [NELL-995](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/NELL-995)      | 75,492      |   149,678 |   200 |
| [FB15k-237](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/FB15K237)      | 14,541 | 310,116 | 237 |
| [YAGO3-10](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks/YAGO3-10)      | 123,182 |   1,179,040 |   37 |

## Getting Start

1. Clone Repository: Clone this repository to your local machine.

   ```python
   git clone https://github.com/rutgerswiselab/KG-LLM.git
   ```

2. Download Data: Download the dataset from the previously mentioned library.

3. Place Files: Put the downloaded dataset in the `preprocess` folder.

### Preprocess Data

1. Run Script: Open a terminal or command prompt, navigate to the directory containing the script and files, and run the following command:
   ```python
   python preprocess.py
   ```
   - Check Output: After running the script, you should find some new CSV files containing the preprocessed data in the same directory.
2. Split Data:
   ```python
   python split_data.py
   python KGLLM_preprocess.py
   ```
   - Check Output: After running the script, you should find the training data, validation data, testing data.

### Finetune
Three distinct LLMs are utilized: Flan-T5-Large, LlaMa2-7B, and Gemma-7B.

| Model  | #Parameter | #Maximum Token  | #Technique  |
| ------------- |:-------------:| -----:| -----:|
| [Flan-T5-Large](https://huggingface.co/google/flan-t5-large)  | 783M | 512 | Global Fine-tune |
| [LLaMa2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)      | 7B      |   4096 |   4bit-Lora |
| [Gemma-7B](https://huggingface.co/google/gemma-7b)      | 7B      |   4096 |   4bit-Lora |


1. Get the Data Ready

  > Make sure your training, validation, testing data is ready to use

2. Train the model

  > Open the `train.py` script in `train` folder and modify all hyperparameters as you like. Here are the parameters you can modify:
```python
per_device_train_batch_size=8,
per_device_eval_batch_size=8,
gradient_accumulation_steps=4,
warmup_steps=2,
weight_decay=0.01,
num_train_epochs=5,
learning_rate=2e-4,
fp16=True,
logging_steps=1,
output_dir="outputs",
optim="paged_adamw_8bit",
```
  > When running the train.py, it takes several arguments:
```python
"--model_name", type=str, default="flan-t5", help="which model: flan-t5, llama2, gemma"
"--train_file", type=str, default=r"train_data.csv", help="Path to the train CSV file"
"--valid_file", type=str, default=r"val_data.csv", help="Path to the validation CSV file"
"--entity_file", type=str, default=r"entity2id.txt", help="Path to the entity2id.txt file"
"--relation_file", type=str, default=r"relation2id.txt", help="Path to the relation2id.txt file"
```
  > Start Finetune:
```python
   python train/train.py --model_name llama2 --train_file train_data.csv --valid_file val_data.csv --entity_file WF/entity2id.txt --relation_file WF/relation2id.txt
```

3. Model Training Completed

  > After training, the trained model checkpoint files will be generated in the specified output_dir directory.

### Test

1. Modify Test File
   
  > Open the `test` folder and test script file, for example (test_link_icl.py)

  > test_file: Modify this parameter to specify the path to the test file.

  > model: Modify this parameter to specify the path to the trained model checkpoint directory.

2. Run the Test Script
   
  > Run the test script (test.py) in the command line and wait for the model testing to complete.
  
Example command:
```python
   python test_link_icl.py
```

3. View Accuracy
  > After testing, the script will output the accuracy of the model on the test dataset.
  
Example output:
```python
   AUC: 0.95
   F1: 0.93
```
