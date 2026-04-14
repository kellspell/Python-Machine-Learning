from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainningArguments
from datasets import load_dataset

# load the dataset
dataset = load_dataset("ag_news")

# Load the Roberta Tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4 )

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare dataset for training
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets = tokenized_datasets.rename_columns({'label': 'labels'})
tokenized_datasets.set_format('torch')

train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']

# Training parameters
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500   
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=test_dataset,
    processing_class=tokenizer
)

trainer.train()

result = trainer.evaluate()
print("Evaluation results: ", results)