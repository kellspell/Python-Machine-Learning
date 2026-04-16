import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, TrainingArguments, Trainer



# Load PubMed 20k RCT dataset
dataset = load_dataset('pubmed_rct', '20k_rct')
print(dataset['train'][0])

# Pre-Process the dataset or data
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# call the function above
tokenizer_datasets = dataset.map(preprocess_data, batched=True)
tokenizer_datasets = tokenizer_datasets.rename_column('label', 'labels') # changing a column name
tokenizer_datasets.set_format('torch')

# Model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# if you want to use a different dataset
# model = AutoModelForSequenceClassification.from_pretrained('dmis-lab/biobert-base-uncased-v1.1', num_labels=5)

# Nest steps is to fine-tune the model
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenizer_datasets['train'],
    eval_dataset=tokenizer_datasets['validation'],
    tokenizer=tokenizer
)

trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation results: " , results)

# here its just an example of how to implement data augmentation for text sentences
def augment_text(text):
    synonymous = {'cancer': ['tumor', 'malignacy'], 'study': ['research', 'experiment']}
    words = text.split()
    new_words = [random.choice(synonymous[word]) if word in synonymous else word for word in words]
    return " ".join(new_words)

augmented_data = [augment_text(sample['text']) for sample in dataset['train']]
print(augmented_data[:5])


 