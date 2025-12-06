"""
Part 5 Extended Analysis - Calibration & Question Difficulty
Analyzes ALL 159 questions for calibration and difficulty patterns
Usage: python extended_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_all_evaluations():
    """Load all evaluation results and questions"""
    
    systems = ['None_gpt4omini', 'Azure_gpt4omini', 'Local_gpt4omini', 
               'Azure_flant5', 'Local_flant5']
    
    # Load questions
    questions_df = pd.read_csv('data/question.tsv', sep='\t', header=None, 
                               names=['question', 'type'])
    
    # Load predictions and evaluations for all systems
    all_data = []
    
    for system in systems:
        # Load predictions
        pred_file = f'output/prediction/{system}.tsv'
        pred_df = pd.read_csv(pred_file, sep='\t', header=None)
        predictions = pred_df[0].tolist()
        
        # Load evaluation results
        eval_file = f'output/evaluation/{system}.tsv'
        eval_df = pd.read_csv(eval_file, sep='\t', header=None,
                             names=['llm_score', 'exact_match', 'f1_score'])
        
        # Combine with questions
        for idx in range(len(questions_df)):
            all_data.append({
                'system': system,
                'question_idx': idx + 1,
                'question': questions_df.loc[idx, 'question'],
                'question_type': questions_df.loc[idx, 'type'],
                'prediction': predictions[idx],
                'llm_score': eval_df.loc[idx, 'llm_score'],
                'exact_match': eval_df.loc[idx, 'exact_match'],
                'f1_score': eval_df.loc[idx, 'f1_score']
            })
    
    return pd.DataFrame(all_data), questions_df


def calibration_analysis(eval_data, output_dir):
    """
    Analyze if LLM scores are well-calibrated with actual performance
    ALL 795 predictions analyzed
    """
    print("\n" + "="*60)
    print("EXTENDED ANALYSIS 1: CALIBRATION ANALYSIS")
    print("Analyzing ALL 795 predictions (159 questions × 5 systems)")
    print("="*60)
    
    # Group by LLM score and compute average metrics
    calibration_results = []
    
    for llm_score in range(1, 6):
        score_data = eval_data[eval_data['llm_score'] == llm_score]
        
        if len(score_data) == 0:
            continue
        
        calibration_results.append({
            'llm_score': llm_score,
            'count': len(score_data),
            'avg_exact_match': score_data['exact_match'].mean(),
            'avg_f1_score': score_data['f1_score'].mean(),
            'exact_match_std': score_data['exact_match'].std(),
            'f1_std': score_data['f1_score'].std()
        })
    
    calib_df = pd.DataFrame(calibration_results)
    calib_df.to_csv(f'{output_dir}/calibration_analysis.csv', index=False)
    
    print(f"\n✓ Calibration Analysis Complete!")
    print(f"  Analyzed {len(eval_data)} predictions")
    print(f"  Saved to: {output_dir}/calibration_analysis.csv\n")
    
    print("Calibration Results (Are high LLM scores trustworthy?):")
    print(calib_df.to_string(index=False))
    
    # Compute correlation
    corr_em = eval_data['llm_score'].corr(eval_data['exact_match'])
    corr_f1 = eval_data['llm_score'].corr(eval_data['f1_score'])
    
    print(f"\nCorrelation with Exact Match: {corr_em:.3f}")
    print(f"Correlation with F1 Score: {corr_f1:.3f}")
    
    # Create calibration plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: LLM Score vs Exact Match
    axes[0].plot(calib_df['llm_score'], calib_df['avg_exact_match'], 
                 marker='o', linewidth=2, markersize=10, color='steelblue')
    axes[0].set_xlabel('LLM Score', fontsize=12)
    axes[0].set_ylabel('Average Exact Match Rate', fontsize=12)
    axes[0].set_title('Calibration: LLM Score vs Exact Match\n(All 795 predictions)', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(1, 6))
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: LLM Score vs F1
    axes[1].plot(calib_df['llm_score'], calib_df['avg_f1_score'], 
                 marker='o', linewidth=2, markersize=10, color='darkorange')
    axes[1].set_xlabel('LLM Score', fontsize=12)
    axes[1].set_ylabel('Average F1 Score', fontsize=12)
    axes[1].set_title('Calibration: LLM Score vs F1 Score\n(All 795 predictions)', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(1, 6))
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/calibration_plot.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Created: {output_dir}/calibration_plot.png")
    
    plt.close()
    
    return calib_df


def question_difficulty_analysis(eval_data, questions_df, output_dir):
    """
    Analyze which questions are universally hard/easy across ALL systems
    ALL 159 questions analyzed
    """
    print("\n" + "="*60)
    print("EXTENDED ANALYSIS 2: QUESTION DIFFICULTY ANALYSIS")
    print("Analyzing ALL 159 questions across all 5 systems")
    print("="*60)
    
    # For each question, compute average performance across all systems
    question_difficulty = []
    
    for q_idx in range(1, 160):  # 1-indexed
        q_data = eval_data[eval_data['question_idx'] == q_idx]
        
        question_difficulty.append({
            'question_idx': q_idx,
            'question': q_data.iloc[0]['question'],
            'question_type': q_data.iloc[0]['question_type'],
            'avg_llm_score': q_data['llm_score'].mean(),
            'avg_exact_match': q_data['exact_match'].mean(),
            'avg_f1_score': q_data['f1_score'].mean(),
            'llm_std': q_data['llm_score'].std(),
            'systems_failed': (q_data['llm_score'] < 3).sum(),  # How many systems scored < 3
            'systems_perfect': (q_data['llm_score'] == 5).sum(),  # How many systems got 5
            'is_universally_hard': (q_data['llm_score'] < 3).sum() >= 4,  # 4+ systems failed
            'is_universally_easy': (q_data['llm_score'] == 5).sum() >= 4   # 4+ systems perfect
        })
    
    diff_df = pd.DataFrame(question_difficulty)
    diff_df.to_csv(f'{output_dir}/question_difficulty_all.csv', index=False)
    
    # Get hardest 20 questions (lowest avg LLM score)
    hardest = diff_df.nsmallest(20, 'avg_llm_score')[
        ['question_idx', 'question', 'question_type', 'avg_llm_score', 
         'avg_exact_match', 'avg_f1_score', 'systems_failed']
    ].copy()
    
    # Truncate long questions
    hardest['question_short'] = hardest['question'].apply(
        lambda x: x[:150] + "..." if len(x) > 150 else x
    )
    
    hardest.to_csv(f'{output_dir}/hardest_20_questions.csv', index=False)
    
    # Get easiest 20 questions (highest avg LLM score)
    easiest = diff_df.nlargest(20, 'avg_llm_score')[
        ['question_idx', 'question', 'question_type', 'avg_llm_score', 
         'avg_exact_match', 'avg_f1_score', 'systems_perfect']
    ].copy()
    
    easiest['question_short'] = easiest['question'].apply(
        lambda x: x[:150] + "..." if len(x) > 150 else x
    )
    
    easiest.to_csv(f'{output_dir}/easiest_20_questions.csv', index=False)
    
    print(f"\n✓ Question Difficulty Analysis Complete!")
    print(f"  Analyzed all 159 questions")
    print(f"  Saved complete analysis to: {output_dir}/question_difficulty_all.csv")
    print(f"  Saved hardest 20 to: {output_dir}/hardest_20_questions.csv")
    print(f"  Saved easiest 20 to: {output_dir}/easiest_20_questions.csv")
    
    # Summary statistics
    universally_hard = diff_df[diff_df['is_universally_hard'] == True]
    universally_easy = diff_df[diff_df['is_universally_easy'] == True]
    
    print(f"\nDifficulty Summary:")
    print(f"  Universally Hard (4+ systems failed): {len(universally_hard)} questions")
    print(f"  Universally Easy (4+ systems perfect): {len(universally_easy)} questions")
    print(f"  Average difficulty (LLM score): {diff_df['avg_llm_score'].mean():.2f}")
    
    print(f"\nTop 10 Hardest Questions:")
    for idx, row in hardest.head(10).iterrows():
        print(f"  Q{row['question_idx']}: {row['question_short'][:100]}")
        print(f"    Avg LLM: {row['avg_llm_score']:.2f}, Failed by {row['systems_failed']}/5 systems\n")
    
    # Create difficulty distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Difficulty histogram
    axes[0].hist(diff_df['avg_llm_score'], bins=20, color='skyblue', edgecolor='black')
    axes[0].axvline(diff_df['avg_llm_score'].mean(), color='red', 
                    linestyle='--', linewidth=2, label=f'Mean: {diff_df["avg_llm_score"].mean():.2f}')
    axes[0].set_xlabel('Average LLM Score Across Systems', fontsize=12)
    axes[0].set_ylabel('Number of Questions', fontsize=12)
    axes[0].set_title('Question Difficulty Distribution\n(All 159 questions)', 
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Difficulty by question type
    type_difficulty = diff_df.groupby('question_type')['avg_llm_score'].mean().sort_values()
    axes[1].barh(type_difficulty.index, type_difficulty.values, color='lightcoral')
    axes[1].set_xlabel('Average LLM Score', fontsize=12)
    axes[1].set_ylabel('Question Type', fontsize=12)
    axes[1].set_title('Average Difficulty by Question Type\n(53 questions per type)', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xlim([0, 5])
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/difficulty_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Created: {output_dir}/difficulty_distribution.png")
    
    plt.close()
    
    return diff_df, hardest, easiest


def statistical_significance_test(eval_data, output_dir):
    """
    Perform statistical tests to check if system differences are significant
    """
    print("\n" + "="*60)
    print("BONUS: STATISTICAL SIGNIFICANCE TESTING")
    print("="*60)
    
    systems = ['None_gpt4omini', 'Azure_gpt4omini', 'Local_gpt4omini', 
               'Azure_flant5', 'Local_flant5']
    
    results = []
    
    # Pairwise t-tests
    for i in range(len(systems)):
        for j in range(i+1, len(systems)):
            sys1 = systems[i]
            sys2 = systems[j]
            
            sys1_scores = eval_data[eval_data['system'] == sys1]['llm_score']
            sys2_scores = eval_data[eval_data['system'] == sys2]['llm_score']
            
            t_stat, p_value = stats.ttest_ind(sys1_scores, sys2_scores)
            
            results.append({
                'system_1': sys1,
                'system_2': sys2,
                'sys1_mean': sys1_scores.mean(),
                'sys2_mean': sys2_scores.mean(),
                'difference': sys1_scores.mean() - sys2_scores.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_at_0.05': p_value < 0.05,
                'significant_at_0.01': p_value < 0.01
            })
    
    sig_df = pd.DataFrame(results)
    sig_df.to_csv(f'{output_dir}/statistical_significance.csv', index=False)
    
    print(f"\n✓ Statistical Significance Tests Complete!")
    print(f"  Performed {len(sig_df)} pairwise t-tests")
    print(f"  Saved to: {output_dir}/statistical_significance.csv\n")
    
    print("Significant Differences (p < 0.05):")
    sig_only = sig_df[sig_df['significant_at_0.05'] == True]
    for idx, row in sig_only.iterrows():
        print(f"  {row['system_1']} vs {row['system_2']}: "
              f"diff={row['difference']:.2f}, p={row['p_value']:.4f}")
    
    return sig_df


def main():
    print("="*60)
    print("PART 5 EXTENDED ANALYSIS")
    print("Calibration + Question Difficulty")
    print("Analyzing ALL 159 questions")
    print("="*60)
    
    # Create output directory
    output_dir = 'output/analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nLoading all evaluation data...")
    eval_data, questions_df = load_all_evaluations()
    print(f"✓ Loaded {len(eval_data)} predictions (159 questions × 5 systems)")
    
    # Run analyses
    calib_df = calibration_analysis(eval_data, output_dir)
    diff_df, hardest, easiest = question_difficulty_analysis(eval_data, questions_df, output_dir)
    sig_df = statistical_significance_test(eval_data, output_dir)
    
    print("\n" + "="*60)
    print("✅ EXTENDED ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nNew files created in {output_dir}/:")
    print("\nCSV Files:")
    print("  - calibration_analysis.csv")
    print("  - question_difficulty_all.csv (ALL 159 questions)")
    print("  - hardest_20_questions.csv")
    print("  - easiest_20_questions.csv")
    print("  - statistical_significance.csv")
    print("\nVisualization Files:")
    print("  - calibration_plot.png")
    print("  - difficulty_distribution.png")
    print("\n" + "="*60)
    print("TOTAL PART 5 OUTPUT FILES: 16")
    print("  - 11 from original analysis")
    print("  - 5 new CSV files")
    print("  - 2 new charts")
    print("="*60)
    print("\nData coverage: 100% (ALL 159 questions)")
    print("="*60)


if __name__ == "__main__":
    main()
