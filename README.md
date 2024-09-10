# Fine-Tuning Meta-Llama-3-8B (Quantized Version) with Local Caching and Metrics Logging

This repository provides a comprehensive script for fine-tuning the quantized Meta-Llama-3-8B model. The script takes advantage of local caching on an external SSD to prevent storage limitations on the main drive. It ensures an efficient fine-tuning process by leveraging the SSD's high-speed read/write capabilities. Additionally, the script is equipped with detailed logging and metrics tracking to thoroughly monitor the fine-tuning performance and system resource usage.

The hardware setup used for testing includes:
- **Processor**: AMD Ryzen 7 1800X Eight-Core Processor, 3.60 GHz
- **Graphics Card**: NVIDIA GeForce RTX 3060
- **Memory**: 32.0 GB RAM
- **Operating System**: Windows on a Dell Inspiron machine

### Why Quantized Meta-Llama-3-8B?

Using a quantized version of the Meta-Llama-3-8B model offers several advantages:
- **Reduced Computational Overhead**: Quantization reduces the numerical precision of model weights, which decreases both computational load and memory usage. This makes it feasible to run fine-tuning tasks on less powerful machines while still maintaining performance efficiency.
- **Faster Execution**: Lower precision calculations are faster to compute, leading to quicker model training and inference.
- **Resource Efficiency**: Deploying large language models in resource-constrained environments becomes possible without a significant loss in accuracy.
- **Energy Efficiency**: Reduced computational requirements lead to lower energy consumption, which is beneficial for both cost and environmental impact.

### Key Features

- **Local SSD Caching**: Utilizes an external SSD for caching to bypass main drive storage constraints, ensuring a smoother fine-tuning experience.
- **Detailed Logging and Metrics**: Provides comprehensive logs and metrics tracking, including training loss, evaluation loss, learning rate, and time taken for each epoch.
- **Performance Analytics**: Generates CSV files and plots to help users analyze the performance of the model over time, aiding in better understanding and further tuning.

## Prerequisites

Before running the script, ensure that you have the following installed:
- Python 3.7 or later
- The required Python packages (`torch`, `transformers`, `datasets`, `peft`, `trl`, `matplotlib`, `python-dotenv`, `csv`)
- A Hugging Face account and an access token

## Environment Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/your_repository.git
    cd your_repository
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv llama3_venv
    source llama3_venv/bin/activate   # On Windows use `llama3_venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```bash
    pip install torch transformers datasets peft trl matplotlib python-dotenv
    ```

4. **Set up the Hugging Face token and environment variables:**

    Create a `.env` file in the project directory with the following content:
    ```env
    HUGGINGFACE_TOKEN_LLAMA=your_hugging_face_token_here
    ```

## Script Details

### fine_tune_llama.py

This script fine-tunes the Meta-Llama-3-8B model and logs detailed metrics to monitor performance. It performs the following tasks:

1. **Loads environment variables and sets up local caching** to prevent storage issues on the main drive.
2. **Authenticates** with Hugging Face using the provided token.
3. **Loads and preprocesses the dataset** to prepare it for training.
4. **Configures and fine-tunes the model** with LoRA tuning.
5. **Logs key metrics** (training loss, evaluation loss, learning rate, time taken) to a CSV file.
6. **Generates inference results** and saves them along with training metrics.
7. **Plots training metrics** using `matplotlib`.

### Usage Instructions

To run the script, use the following command:
```bash
python fine_tune_llama.py
```

## Output Files

After the script completes, the following files will be available in the specified output directory (`F:/output_llama3_qlora`):

- `training_metrics.csv`: Contains epoch-wise training and evaluation metrics.
- `train_results.txt`: Contains the total training time and results summary.
- `inference_log.txt`: Contains the generated inference result.
- `training_metrics.png`: Plot of training and evaluation loss over epochs.

## How to Interact with and Use the Stats

### View Training Metrics
The `training_metrics.csv` file provides detailed metrics of the training process for each epoch. You can open this file with any CSV viewer (such as Excel or Google Sheets) to analyze the training and evaluation metrics.

### Review Training Results
The `train_results.txt` file contains the total training time and a summary of the results. This file provides a quick overview of how the training process performed.

### Examine Inference Result
The `inference_log.txt` file contains the model's response to a sample prompt provided during inference. This file helps you to validate the fine-tuned model’s performance in generating text.

### Visualize Metrics
The `training_metrics.png` image is a plot that shows the training and evaluation loss over epochs. This visualization helps to understand the model's performance and training stability over time.

### Example of Inference Interaction

To generate text using the fine-tuned model, you need to load the model and tokenizer, then use the pipeline functionality from Hugging Face's Transformers library. Here’s a sample script to interact with the fine-tuned model:

```python
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load environment variables from .env file
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN_LLAMA")

# Load the fine-tuned model and tokenizer
output_dir = "F:/output_llama3_qlora"
tokenizer = AutoTokenizer.from_pretrained(output_dir, token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(output_dir, token=huggingface_token, torch_dtype=torch.float16).to("cuda")

# Create a text generation pipeline
generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Define the prompt
prompt = "User: I'm feeling really anxious about my upcoming exams. Assistant:"

# Generate text
response = generation_pipeline(prompt, max_new_tokens=150)
print(response[0]['generated_text'])
```

## Post-Fine-Tuning Evaluation

### post_finetune_eval.py

This script evaluates the performance of the fine-tuned Meta-Llama-3-8B model by comparing its responses to a set of predefined prompts against those generated by the original pre-trained model. The script performs the following tasks:

1. **Authenticates with Hugging Face**: Verifies the user using the Hugging Face token.
2. **Loads both the original pre-trained model and the fine-tuned model**: Sets up both models for comparison.
3. **Generates responses**: Produces outputs for each prompt using both the fine-tuned and original models.
4. **Calculates cosine similarity**: Uses embeddings from the `sentence-transformers` model to compute the similarity between responses from each model.
5. **Logs and saves comparisons**: Logs the comparison results, including cosine similarity scores, to a text file.

### Usage Instructions

To run the evaluation script, use the following command:
```bash
python post_finetune_eval.py
```

### Output Files

After the evaluation script completes, the following file will be available in the specified output directory (`F:/output_llama3_qlora`):

- `comparison_log.txt`: Contains prompt-wise comparisons of responses from the original and fine-tuned models, along with cosine similarity scores.

### Example Prompts for Evaluation

The script includes several sample prompts focused on mental health conversations and general assistance:
- "User: I'm feeling really anxious about my upcoming exams. Assistant:"
- "User: Can you tell me a good video game released in 2020? Assistant:"
- "User: What's the best way to prepare for a programming interview? Assistant:"
- Additional prompts covering various aspects of mental health and everyday queries.

## Hardware and Performance Notes

This script was tested using a Dell Inspiron with an AMD Ryzen 7 1800X Eight-Core Processor, 32 GB RAM, and an RTX 3060 GPU. While this hardware setup is considered underpowered for large-scale deep learning tasks, it was still capable of completing the fine-tuning process. For optimal performance, using a machine with more powerful GPU capabilities is recommended.

## Acknowledgements

This project leverages the powerful capabilities of Hugging Face's Transformers library and the fine-tuning techniques provided by `peft` and `trl` libraries. For more information on Hugging Face and the Meta-Llama-3-8B model, please visit [Hugging Face](https://huggingface.co).