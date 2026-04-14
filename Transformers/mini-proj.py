from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import load_dataset
import evaluate

# Load dataset for summarization
dataset = load_dataset('cnn_dailymail', '3.0.0')
print(dataset['train'][0])

# for translation
# dataset = load_dataset('wm14', 'en-fr')


# Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Tokenize the dataset for summarization
def tokenize_function(examples):
    inputs = ['summarize: ' + doc for doc in examples['article']]
    model_inputs = tokenizer(inputs, truncation=True, max_length=512)

    labels = tokenizer(text_target=examples['highlights'], max_length=150, truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)   
    

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

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

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

sample_txt = 'The transformer has revolutionize NLP by enbling parallel processing sequence.'
inputs = tokenizer('summarize: ' + sample_txt, return_tensors='pt', max_length=512, truncation=True)
outputs = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)

print('Generated Summary: ', tokenizer.decode(outputs[0], skip_special_token=True))
metric = evaluate.load('rouge') # sacrenleu
predictions = outputs['generated_text']
references = dataset['validation']['highlights']

results = metric.compute(predictions=predictions, references=references)

result = trainer.evaluate()
print("Evaluation results: ", results)