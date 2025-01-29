import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
import evaluate
from datasets import load_metric

# Load metrics
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
sari = load_metric('sari')

# Load models
base_model_name = "t5-small"
fine_tuned_model_path = "./simplification_model"

base_tokenizer = T5Tokenizer.from_pretrained(base_model_name)
base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)

ft_tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_path)
ft_model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path)

# Load evaluation data
# Use sample for demonstration
eval_df = pd.read_csv(
    './sensible_complex_simplified_dataset.csv').sample(n=100)


def generate_predictions(model, tokenizer, complex_texts):
    predictions = []
    for text in complex_texts:
        inputs = tokenizer.encode(
            "simplify: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs, max_length=50, num_beams=4, early_stopping=True)
        predictions.append(tokenizer.decode(
            outputs[0], skip_special_tokens=True))
    return predictions


# Generate predictions
base_preds = generate_predictions(
    base_model, base_tokenizer, eval_df['Complex_Text'].tolist())
ft_preds = generate_predictions(
    ft_model, ft_tokenizer, eval_df['Complex_Text'].tolist())
# BLEU requires list of lists
references = [[ref] for ref in eval_df['Simplified_Text'].tolist()]


def calculate_metrics(predictions, references, sources):
    # BLEU
    bleu_results = bleu.compute(predictions=predictions, references=references)

    # ROUGE
    rouge_results = rouge.compute(
        predictions=predictions, references=references)

    # SARI (requires original sources)
    sari_results = sari.compute(
        sources=sources, predictions=predictions, references=references)

    return {
        'BLEU': bleu_results['bleu'],
        'ROUGE-1': rouge_results['rouge1'],
        'ROUGE-2': rouge_results['rouge2'],
        'ROUGE-L': rouge_results['rougeL'],
        'SARI': sari_results['sari']
    }


# Calculate metrics
base_metrics = calculate_metrics(
    base_preds, references, eval_df['Complex_Text'].tolist())
ft_metrics = calculate_metrics(
    ft_preds, references, eval_df['Complex_Text'].tolist())

# Create comparison table
metrics_df = pd.DataFrame({
    'Metric': ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'SARI'],
    'Base Model': [base_metrics[m] for m in ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'SARI']],
    'Fine-Tuned Model': [ft_metrics[m] for m in ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'SARI']]
})

print("\nModel Performance Comparison:")
print(metrics_df.to_markdown(index=False))
