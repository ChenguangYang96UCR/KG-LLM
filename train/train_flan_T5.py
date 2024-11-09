
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

# Load the FLAN-T5 model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large', model_max_length=512)

# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        # Get the input and output sequences
        input_sequence = self.data.iloc[idx]['train_input_text']
        output_sequence = self.data.iloc[idx]['train_output_text']

        # Add task-specific prefix to input sequence
        # input_sequence = 'is first node connect with last node: ' + input_sequence

        # Encode the input and output sequences using the T5 tokenizer
        input_encoding = tokenizer(input_sequence, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        output_encoding = tokenizer(output_sequence, padding='max_length', max_length=512, truncation=True, return_tensors='pt')

        # Get the input IDs, attention mask, and label IDs from the encodings
        input_ids = input_encoding['input_ids'].squeeze()
        attention_mask = input_encoding['attention_mask'].squeeze()
        label_ids = output_encoding['input_ids'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label_ids': label_ids}

    def __len__(self):
        return len(self.data)

# Define the data collator function
def data_collator(batch):
    input_ids = torch.stack([example['input_ids'] for example in batch])
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    label_ids = torch.stack([example['label_ids'] for example in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label_ids}

# Load the data
preprocess_data = pd.read_csv('train.csv')

# Split the data into train and validation sets
train_data = preprocess_data.sample(frac=0.8, random_state=1)
val_data = preprocess_data.drop(train_data.index)

# Create the datasets
train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
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
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()
