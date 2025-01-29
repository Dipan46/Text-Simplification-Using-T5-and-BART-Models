import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import sacrebleu

def compute_sari(orig, simplified, references):
    """
    Custom implementation of SARI score.
    """
    add_precision = sacrebleu.sentence_bleu(simplified, [ref for ref in references]).score
    keep_precision = sacrebleu.sentence_bleu(simplified, [orig]).score
    delete_precision = sacrebleu.sentence_bleu(orig, [simplified]).score
    sari_score = (add_precision + keep_precision + delete_precision) / 3
    return sari_score

def compute_scores(reference, model_output):
    # BLEU Score
    bleu_score = sentence_bleu([reference.split()], model_output.split())

    # ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, model_output)

    # SARI Score
    sari_score = compute_sari(reference, model_output, [reference])

    return {
        'BLEU': bleu_score,
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure,
        'SARI': sari_score
    }

# Example Reference and Model Outputs
reference_text = "The quick brown fox jumps over the lazy dog."
t5_simplified = "The fast brown fox leaps over the sleepy dog."
bart_simplified = "The quick fox jumps over the lazy dog."

# Compute Scores
t5_scores = compute_scores(reference_text, t5_simplified)
bart_scores = compute_scores(reference_text, bart_simplified)

# Plot Comparison Graph
metrics = list(t5_scores.keys())
t5_values = list(t5_scores.values())
bart_values = list(bart_scores.values())

x = range(len(metrics))

plt.figure(figsize=(8, 5))
plt.bar(x, t5_values, width=0.4, label='T5', color='blue', align='center')
plt.bar([i + 0.4 for i in x], bart_values, width=0.4, label='BART', color='green', align='center')

plt.xticks([i + 0.2 for i in x], metrics)
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.title("T5 vs BART Model Evaluation Metrics")
plt.ylim(0, 1)  # Normalize scores between 0 and 1
plt.legend()
plt.show()
