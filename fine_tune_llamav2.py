import os
import time
import torch
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import csv
import matplotlib.pyplot as plt

# Track the start of the entire script execution
script_start_time = time.time()

# Load environment variables from .env file (.env file should contain HUGGINGFACE_TOKEN_LLAMA)
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN_LLAMA")

if huggingface_token is None:
    raise ValueError("[ERROR] Hugging Face token is not set. Please set HUGGINGFACE_TOKEN_LLAMA in the .env file.")

# Set up environment variables for Hugging Face cache. Logging the cache directory.
cache_dir = 'F:/huggingface_cache'
os.environ['HF_HOME'] = cache_dir
os.environ['HF_HUB_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = f'{cache_dir}/datasets'
os.environ['DISABLE_TELEMETRY'] = 'YES'
print(f"[INFO] Set Hugging Face cache directory to {cache_dir}")

# Authenticate and login to Hugging Face
print("[INFO] Logging into Hugging Face...")
login(token=huggingface_token)
print("[INFO] Login successful")

# Load and prepare the dataset
print("[INFO] Loading dataset...")
df = pd.read_csv('dataset/reddit_text-davinci-002.csv')
dataset = Dataset.from_pandas(df)

def format_for_chatbot(row):
    row["text"] = f"User: {row['prompt']} Assistant: {row['completion']}"
    return row

print("[INFO] Formatting dataset for chatbot...")
dataset = dataset.map(format_for_chatbot)
dataset = dataset.train_test_split(test_size=0.1)
print("[INFO] Dataset split into training and testing sets")

# Model and tokenizer setup
model_id = "meta-llama/Meta-Llama-3-8B"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token, cache_dir=cache_dir)

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    print("[INFO] Tokenizer does not have a pad token. Adding pad token...")
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

print("[INFO] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    token=huggingface_token,
    cache_dir=cache_dir,
    device_map='auto',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Configure LoRA
print("[INFO] Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

model = get_peft_model(model, lora_config)

# Ensure the model is correctly prepared for kbit training
print("[INFO] Preparing model for kbit training...")
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Ensure use_cache is set to False to avoid issues with gradient checkpointing

# Training setup
output_dir = "F:/output_llama3_qlora"
os.makedirs(output_dir, exist_ok=True)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    logging_steps=10,
    eval_strategy="steps",
    save_steps=50,
    save_total_limit=1,
    fp16=False,
    logging_dir=f"{output_dir}/logs",
    report_to=[]  # Disable any reporting
)
print(f"[INFO] Training arguments set. Output directory: {output_dir}")

# Initialize CSV logging
metrics_file = f"{output_dir}/training_metrics.csv"
with open(metrics_file, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "training_loss", "eval_loss", "learning_rate", "time_taken"])
    writer.writeheader()

class MetricsLogger(TrainerCallback):
    def __init__(self, metrics_file):
        self.metrics_file = metrics_file
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("[INFO] Training started")

    def on_log(self, args, state, control, logs=None, **kwargs):
        epoch = logs.get("epoch", state.epoch if state else -1)
        log_data = {
            "epoch": epoch,
            "training_loss": logs.get("loss", float('nan')),
            "eval_loss": logs.get("eval_loss", float('nan')),
            "learning_rate": logs.get("learning_rate", float('nan')),
            "time_taken": time.time() - self.start_time
        }
        with open(self.metrics_file, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            writer.writerow(log_data)
        print(f"[INFO] Epoch {epoch}: training_loss = {logs.get('loss', 'N/A')}, eval_loss = {logs.get('eval_loss', 'N/A')}, "
              f"learning_rate = {logs.get('learning_rate', 'N/A')}, time_taken = {time.time() - self.start_time}")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=512,
    dataset_text_field="text",
    packing=False,
    callbacks=[MetricsLogger(metrics_file)]  # Logging metrics with MetricsLogger
)

# Fine-tune the model - Verbose logging added
start_time = time.time()
try:
    print("[INFO] Beginning fine-tuning...")
    train_result = trainer.train()
finally:
    end_time = time.time()
    print(f"[INFO] Training completed in {end_time - start_time:.2f} seconds")

# Save the fine-tuned model
print("[INFO] Saving the fine-tuned model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save training time and results
with open(f"{output_dir}/train_results.txt", "w") as f:
    f.write(f"Training time: {end_time - start_time}\n")
    f.write(f"Training results: {train_result}\n")
print("[INFO] Training results saved")

# Run inference to evaluate the model - Verbose logging added
print("[INFO] Running inference to evaluate the model...")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "User: I'm feeling really anxious about my upcoming exams. Assistant:"
result = pipe(prompt, max_length=150, num_return_sequences=1)
print("[INFO] Inference result:")
print(result[0]['generated_text'])

# Save result to a log file
with open(f"{output_dir}/inference_log.txt", "w") as f:
    f.write(result[0]['generated_text'])
print("[INFO] Inference result saved")

# Plot training metrics
print("[INFO] Plotting training metrics...")
metrics_df = pd.read_csv(metrics_file)
plt.figure(figsize=(10, 5))
plt.plot(metrics_df["epoch"], metrics_df["training_loss"], label="Training Loss")
if "eval_loss" in metrics_df.columns:
    plt.plot(metrics_df["epoch"], metrics_df["eval_loss"], label="Evaluation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss Over Epochs")
plt.legend()
plt.savefig(f"{output_dir}/training_metrics.png")
print(f"[INFO] Training and evaluation metrics plot saved to {output_dir}/training_metrics.png")

# Track the end of the entire script execution
script_end_time = time.time()
total_script_time = script_end_time - script_start_time

# Summary section
print("\n=== SUMMARY ===")
print(f"Total script execution time: {total_script_time:.2f} seconds")
print(f"Training time: {end_time - start_time:.2f} seconds")
print(f"Output directory: {output_dir}")
print(f"Number of training examples: {len(dataset['train'])}")
print(f"Number of evaluation examples: {len(dataset['test'])}")

# Additional metrics from training
training_metrics = pd.read_csv(metrics_file)
latest_metrics = training_metrics.iloc[-1]
print("\n=== FINAL TRAINING METRICS ===")
print(f"Epoch: {latest_metrics['epoch']}")
print(f"Training Loss: {latest_metrics['training_loss']}")
print(f"Evaluation Loss: {latest_metrics['eval_loss']}")
print(f"Learning Rate: {latest_metrics['learning_rate']}")
print(f"Time Taken: {latest_metrics['time_taken']}")