"""
Evaluation Script - Evaluate QA system predictions
Usage: python evaluate.py --file <prediction_file>
"""

import argparse
import pandas as pd
import os
import re
from typing import List, Tuple
from openai import OpenAI
from tqdm import tqdm
import time


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    text = text.lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation at the end
    text = re.sub(r'[.,!?;:]+$', '', text)
    return text


def compute_exact_match(prediction: str, gold_answers: List[str]) -> int:
    """
    Compute exact match score (1 if prediction matches any gold answer, 0 otherwise)
    """
    pred_norm = normalize_text(prediction)
    
    for gold in gold_answers:
        if not gold.strip():  # Skip empty gold answers
            continue
        gold_norm = normalize_text(gold)
        if pred_norm == gold_norm:
            return 1
    
    return 0


def compute_f1_score(prediction: str, gold_answers: List[str]) -> float:
    """
    Compute token-level F1 score (max F1 across all gold answers)
    """
    pred_tokens = set(normalize_text(prediction).split())
    
    if not pred_tokens:
        return 0.0
    
    max_f1 = 0.0
    
    for gold in gold_answers:
        if not gold.strip():
            continue
        
        gold_tokens = set(normalize_text(gold).split())
        
        if not gold_tokens:
            continue
        
        # Compute precision and recall
        common = pred_tokens & gold_tokens
        
        if not common:
            f1 = 0.0
        else:
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        
        max_f1 = max(max_f1, f1)
    
    return max_f1


def llm_as_judge(question: str, prediction: str, gold_answers: List[str], 
                 client: OpenAI, model: str) -> int:
    """
    Use LLM to judge answer quality on a scale of 1-5
    """
    # Format gold answers
    gold_text = " OR ".join([f'"{g}"' for g in gold_answers if g.strip()])
    
    prompt = f"""You are evaluating answers to questions about Basketball in Africa.

Question: {question}

Correct Answer(s): {gold_text}

Student's Answer: {prediction}

Rate the student's answer on a scale of 1-5:
- 5: Perfect answer, matches the correct answer exactly or equivalently
- 4: Very good answer, captures the main point with minor differences
- 3: Acceptable answer, partially correct but missing key information
- 2: Poor answer, contains some relevant information but mostly incorrect
- 1: Wrong answer, completely incorrect or irrelevant

Respond with ONLY a single number (1, 2, 3, 4, or 5). No explanation."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a fair and accurate evaluator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )
        
        score_text = response.choices[0].message.content.strip()
        # Extract first digit
        match = re.search(r'[1-5]', score_text)
        if match:
            return int(match.group())
        else:
            return 3  # Default to middle score if parsing fails
    
    except Exception as e:
        print(f"Error in LLM judge: {e}")
        return 3  # Default to middle score on error


def load_gold_answers(filepath: str) -> List[List[str]]:
    """Load gold answers from TSV file"""
    df = pd.read_csv(filepath, sep='\t', header=None)
    
    gold_answers = []
    for idx, row in df.iterrows():
        # Each row has multiple possible answers (tab-separated)
        answers = [str(val) for val in row.values if pd.notna(val) and str(val).strip()]
        gold_answers.append(answers)
    
    return gold_answers


def load_predictions(filepath: str) -> Tuple[List[str], List[str]]:
    """Load predictions from TSV file"""
    df = pd.read_csv(filepath, sep='\t', header=None)
    
    predictions = df[0].tolist()
    retrieved_docs = df[1].tolist() if len(df.columns) > 1 else [None] * len(predictions)
    
    return predictions, retrieved_docs


def evaluate_system(predictions_file: str, gold_file: str, 
                   api_key: str, base_url: str, output_dir: str):
    """
    Evaluate a QA system's predictions
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = "gpt-4o-mini-2024-07-18"
    
    # Load data
    print(f"Loading predictions from: {predictions_file}")
    predictions, retrieved_docs = load_predictions(predictions_file)
    
    print(f"Loading gold answers from: {gold_file}")
    gold_answers = load_gold_answers(gold_file)
    
    # Load questions for LLM judge
    questions_file = gold_file.replace('answer.tsv', 'question.tsv')
    questions_df = pd.read_csv(questions_file, sep='\t', header=None, names=['question', 'type'])
    questions = questions_df['question'].tolist()
    
    if len(predictions) != len(gold_answers) or len(predictions) != len(questions):
        raise ValueError(f"Mismatch in lengths: {len(predictions)} predictions, {len(gold_answers)} gold answers, {len(questions)} questions")
    
    print(f"\n✓ Loaded {len(predictions)} predictions and gold answers")
    print(f"\nEvaluating with 3 metrics:")
    print("  1. LLM-as-Judge (1-5 scale)")
    print("  2. Exact Match (0 or 1)")
    print("  3. F1 Score (0.0-1.0)")
    print()
    
    # Compute metrics
    llm_scores = []
    exact_matches = []
    f1_scores = []
    
    for idx in tqdm(range(len(predictions)), desc="Evaluating"):
        pred = str(predictions[idx])
        gold = gold_answers[idx]
        question = questions[idx]
        
        # Metric 1: LLM-as-judge
        llm_score = llm_as_judge(question, pred, gold, client, model)
        llm_scores.append(llm_score)
        
        # Metric 2: Exact Match
        em_score = compute_exact_match(pred, gold)
        exact_matches.append(em_score)
        
        # Metric 3: F1 Score
        f1_score = compute_f1_score(pred, gold)
        f1_scores.append(f1_score)
        
        # Small delay to avoid rate limits
        if idx % 20 == 0 and idx > 0:
            time.sleep(1)
    
    # Save results
    system_name = os.path.basename(predictions_file).replace('.tsv', '')
    output_file = os.path.join(output_dir, f"{system_name}.tsv")
    
    # Create output dataframe
    results_df = pd.DataFrame({
        'llm_score': llm_scores,
        'exact_match': exact_matches,
        'f1_score': f1_scores
    })
    
    results_df.to_csv(output_file, sep='\t', header=False, index=False)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {system_name}")
    print(f"{'='*60}")
    print(f"LLM-as-Judge (avg):  {sum(llm_scores)/len(llm_scores):.2f} / 5.0")
    print(f"Exact Match:         {sum(exact_matches)/len(exact_matches)*100:.1f}%")
    print(f"F1 Score (avg):      {sum(f1_scores)/len(f1_scores):.2f}")
    print(f"\n✓ Results saved to: {output_file}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate QA System Predictions")
    parser.add_argument('--file', type=str, required=True,
                        help='Path to predictions TSV file')
    parser.add_argument('--gold', type=str, default='data/answer.tsv',
                        help='Path to gold answers TSV file')
    parser.add_argument('--output', type=str, default='output/evaluation',
                        help='Output directory for evaluation results')
    parser.add_argument('--api-key', type=str, default=None,
                        help='CMU AI Gateway API key')
    parser.add_argument('--base-url', type=str,
                        default='https://ai-gateway.andrew.cmu.edu/',
                        help='API base URL')
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.getenv('CMU_API_KEY')
    
    if not api_key:
        raise ValueError("API key required! Set CMU_API_KEY environment variable or use --api-key")
    
    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)
    
    # Run evaluation
    evaluate_system(args.file, args.gold, api_key, args.base_url, args.output)


if __name__ == "__main__":
    main()
