
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import torch
# import pandas as pd

import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
from datasets import load_dataset, Dataset as HFDataset
import pandas as pd
import argparse
from huggingface_hub import login

seed = 42
set_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#TODO:
# !huggingface-cli login
login(token = "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")


def load_model(model_name, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# def preprocess_batch(batch, tokenizer):
#     return tokenizer(batch["Prompt"], truncation=True, padding='max_length', max_length=tokenizer.model_max_length)

def preprocess_batch(batch, tokenizer):
    max_length = 512  # Set a reasonable max_length for tokenization
    inputs = tokenizer(batch['input_text'], padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    outputs = tokenizer(batch['output_text'], padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    return {
        'input_ids': inputs.input_ids.squeeze(),
        'attention_mask': inputs.attention_mask.squeeze(),
        'labels': outputs.input_ids.squeeze()
    }


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 512
        print(f"Using default max length: {max_length}")
    return max_length

def train(model, tokenizer, train_dataset, valid_dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            num_train_epochs=5,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs


    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    ###

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()

def add_special_tokens(tokenizer, model, entity_file, relation_file):
    with open(entity_file, "r") as ef:
        ef.readline()  # Skip the first line
        entity_ids = [f"entity_{line.strip().split()[0]}" for line in ef]
        node_ids = [f"node_{line.strip().split()[1]}" for line in ef]

    with open(relation_file, "r") as rf:
        rf.readline()  # Skip the first line
        relation_ids = [f"relation_{line.strip().split()[1]}" for line in rf]

    special_tokens = entity_ids + node_ids + relation_ids
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer

# Define the dataset class
# class MyDataset(Dataset):
#     def __init__(self, data_frame, tokenizer):
#         self.data_frame = data_frame
#         self.tokenizer = tokenizer

#     def __getitem__(self, idx):
#         # Get the input and output sequences
#         input_sequence = self.data_frame.iloc[idx]['Prompt']
#         output_sequence = self.data_frame.iloc[idx]['output_text']

#         # Add task-specific prefix to input sequence
#         # input_sequence = 'is first node connect with last node: ' + input_sequence

#         # Encode the input and output sequences using the T5 tokenizer
#         input_encoding = self.tokenizer(input_sequence, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
#         output_encoding = self.tokenizer(output_sequence, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

#         # Get the input IDs, attention mask, and label IDs from the encodings
#         input_ids = input_encoding['input_ids'].squeeze().to(device)
#         attention_mask = input_encoding['attention_mask'].squeeze().to(device)
#         label_ids = output_encoding['input_ids'].squeeze().to(device)

#         return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label_ids': label_ids}

#     def __len__(self):
#         return len(self.data_frame)


def main(model_name, train_file, valid_file, entity_file, relation_file):
    if model_name == "flan-t5":
        # Load the FLAN-T5 model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large').to(device)
        tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large', model_max_length=512)
        tokenizer = add_special_tokens(tokenizer, model, entity_file, relation_file)


        # Define the data collator function
        # def data_collator(batch):
        #     input_ids = torch.stack([example['input_ids'] for example in batch])
        #     attention_mask = torch.stack([example['attention_mask'] for example in batch])
        #     label_ids = torch.stack([example['label_ids'] for example in batch])
        #     return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label_ids}

        # Load the data
        train_data = pd.read_csv(train_file)
        valid_data = pd.read_csv(valid_file)

        train_dataset = HFDataset.from_pandas(train_data)
        valid_dataset = HFDataset.from_pandas(valid_data)

        _preprocessing_function = partial(preprocess_batch, tokenizer=tokenizer)
        train_dataset = train_dataset.map(_preprocessing_function, batched=True)
        valid_dataset = valid_dataset.map(_preprocessing_function, batched=True)

        train_dataset = train_dataset.shuffle(seed=seed)

        # Define the training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,
            learning_rate=3e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy='steps',
            eval_steps=500,
            save_steps=1000,
        )

        # Define the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=lambda data: {
                'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in data]),
                'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in data]),
                'labels': torch.stack([torch.tensor(item['labels']) for item in data]),
            },
            tokenizer=tokenizer,
        )

        # Fine-tune the model
        trainer.train()

    elif model_name == "llama2":

        model_name = "meta-llama/Llama-2-7b-hf"
        bnb_config = create_bnb_config()
        model, tokenizer = load_model(model_name, bnb_config)
        tokenizer = add_special_tokens(tokenizer, model, entity_file, relation_file)

        train_data = pd.read_csv(train_file)
        valid_data = pd.read_csv(valid_file)

        train_dataset = HFDataset.from_pandas(train_data)
        valid_dataset = HFDataset.from_pandas(valid_data)

        _preprocessing_function = partial(preprocess_batch, tokenizer=tokenizer)

        train_dataset = train_dataset.map(_preprocessing_function, batched=True)
        valid_dataset = valid_dataset.map(_preprocessing_function, batched=True)

        train_dataset = train_dataset.shuffle(seed=seed)

        output_dir = "llama2_checkpoint"
        train(model, tokenizer, train_dataset, valid_dataset, output_dir)

    elif model_name == "gemma":
        model_name = "google/gemma-7b"
        bnb_config = create_bnb_config()
        model, tokenizer = load_model(model_name, bnb_config)
        tokenizer = add_special_tokens(tokenizer, model, entity_file, relation_file)

        train_data = pd.read_csv(train_file)
        valid_data = pd.read_csv(valid_file)

        train_dataset = HFDataset.from_pandas(train_data)
        valid_dataset = HFDataset.from_pandas(valid_data)

        _preprocessing_function = partial(preprocess_batch, tokenizer=tokenizer)

        train_dataset = train_dataset.map(_preprocessing_function, batched=True)
        valid_dataset = valid_dataset.map(_preprocessing_function, batched=True)

        train_dataset = train_dataset.shuffle(seed=seed)

        output_dir = "gemma_checkpoint"
        train(model, tokenizer, train_dataset, valid_dataset, output_dir)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="Script for fine-tuning KGLLM models")
    parser.add_argument("--model_name", type=str, default="flan-t5", help="which model: flan-t5, llama2, gemma")
    parser.add_argument("--train_file", type=str, default=r"train_data.csv", help="Path to the train CSV file")
    parser.add_argument("--valid_file", type=str, default=r"val_data.csv", help="Path to the validation CSV file")
    parser.add_argument("--entity_file", type=str, default=r"entity2id.txt", help="Path to the entity2id.txt file")
    parser.add_argument("--relation_file", type=str, default=r"relation2id.txt", help="Path to the relation2id.txt file")
    args = parser.parse_args()

    main(model_name=args.model_name, train_file=args.train_file, valid_file=args.valid_file, entity_file=args.entity_file, relation_file=args.relation_file)
