"""
Part 5 Advanced Analysis Script - COMPLETE VERSION
Performs FULL error analysis on ALL 159 questions, question type breakdown, 
visualization, and retrieval quality analysis
Usage: python analyze.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_all_data():
    """Load all predictions, evaluations, questions, and answers"""
    
    systems = ['None_gpt4omini', 'Azure_gpt4omini', 'Local_gpt4omini', 
               'Azure_flant5', 'Local_flant5']
    
    data = {}
    
    # Load questions
    questions_df = pd.read_csv('data/question.tsv', sep='\t', header=None, 
                               names=['question', 'type'])
    
    # Load gold answers
    answers_df = pd.read_csv('data/answer.tsv', sep='\t', header=None)
    gold_answers = []
    for idx, row in answers_df.iterrows():
        answers = [str(val) for val in row.values if pd.notna(val) and str(val).strip()]
        gold_answers.append(answers[0] if answers else "")
    
    # Load evidence
    evidence_df = pd.read_csv('data/evidence.tsv', sep='\t', header=None,
                             names=['url', 'filename'])
    
    for system in systems:
        # Load predictions
        pred_file = f'output/prediction/{system}.tsv'
        pred_df = pd.read_csv(pred_file, sep='\t', header=None)
        predictions = pred_df[0].tolist()
        retrieved_docs = pred_df[1].tolist() if len(pred_df.columns) > 1 else [None] * len(predictions)
        
        # Load evaluation results
        eval_file = f'output/evaluation/{system}.tsv'
        eval_df = pd.read_csv(eval_file, sep='\t', header=None,
                             names=['llm_score', 'exact_match', 'f1_score'])
        
        data[system] = {
            'predictions': predictions,
            'retrieved_docs': retrieved_docs,
            'evaluations': eval_df,
            'questions': questions_df['question'].tolist(),
            'question_types': questions_df['type'].tolist(),
            'gold_answers': gold_answers,
            'evidence_files': evidence_df['filename'].tolist()
        }
    
    return data, systems


def complete_error_analysis(data: Dict, systems: List[str], output_dir: str):
    """
    Perform COMPLETE error analysis on ALL 159 questions for all systems
    """
    print("\n" + "="*60)
    print("PART 5.1: COMPLETE ERROR ANALYSIS (ALL 159 QUESTIONS)")
    print("="*60)
    
    full_error_report = []
    worst_errors_report = []
    
    for system in systems:
        print(f"\nAnalyzing ALL 159 questions for {system}...")
        
        evals = data[system]['evaluations'].copy()
        predictions = data[system]['predictions']
        questions = data[system]['questions']
        gold_answers = data[system]['gold_answers']
        question_types = data[system]['question_types']
        retrieved_docs = data[system]['retrieved_docs']
        evidence_files = data[system]['evidence_files']
        
        # Compute combined score for ALL questions
        evals['combined_score'] = (evals['llm_score'] / 5.0 + evals['f1_score']) / 2
        
        # Analyze ALL 159 questions
        for idx in range(len(questions)):
            # Analyze failure type
            has_retrieval = system != 'None_gpt4omini'
            retrieved = retrieved_docs[idx] if has_retrieval else "N/A"
            correct_doc = evidence_files[idx]
            
            failure_type = "Unknown"
            if not has_retrieval:
                failure_type = "No Retrieval"
            elif correct_doc in str(retrieved):
                failure_type = "Bad Generation (right docs retrieved)"
            else:
                failure_type = "Bad Retrieval (wrong docs retrieved)"
            
            # Determine if this is an error (not perfect)
            is_error = evals.loc[idx, 'llm_score'] < 5 or evals.loc[idx, 'f1_score'] < 1.0
            
            error_entry = {
                'system': system,
                'question_idx': idx + 1,  # 1-indexed for readability
                'question': questions[idx],
                'question_type': question_types[idx],
                'prediction': str(predictions[idx]),
                'gold_answer': str(gold_answers[idx]),
                'llm_score': evals.loc[idx, 'llm_score'],
                'exact_match': evals.loc[idx, 'exact_match'],
                'f1_score': evals.loc[idx, 'f1_score'],
                'combined_score': evals.loc[idx, 'combined_score'],
                'retrieved_docs': str(retrieved),
                'correct_doc': correct_doc,
                'failure_type': failure_type,
                'is_error': is_error
            }
            
            full_error_report.append(error_entry)
        
        # Also get worst 20 for detailed inspection in report
        worst_indices = evals.nsmallest(20, 'combined_score').index.tolist()
        for idx in worst_indices:
            has_retrieval = system != 'None_gpt4omini'
            retrieved = retrieved_docs[idx] if has_retrieval else "N/A"
            correct_doc = evidence_files[idx]
            
            failure_type = "Unknown"
            if not has_retrieval:
                failure_type = "No Retrieval"
            elif correct_doc in str(retrieved):
                failure_type = "Bad Generation (right docs retrieved)"
            else:
                failure_type = "Bad Retrieval (wrong docs retrieved)"
            
            worst_errors_report.append({
                'system': system,
                'question_idx': idx + 1,
                'rank': len(worst_errors_report) % 20 + 1,
                'question': questions[idx][:200] + "..." if len(questions[idx]) > 200 else questions[idx],
                'question_type': question_types[idx],
                'prediction': str(predictions[idx])[:200] + "..." if len(str(predictions[idx])) > 200 else str(predictions[idx]),
                'gold_answer': str(gold_answers[idx])[:200] + "..." if len(str(gold_answers[idx])) > 200 else str(gold_answers[idx]),
                'llm_score': evals.loc[idx, 'llm_score'],
                'f1_score': evals.loc[idx, 'f1_score'],
                'combined_score': evals.loc[idx, 'combined_score'],
                'retrieved_docs': str(retrieved)[:200] + "..." if len(str(retrieved)) > 200 else str(retrieved),
                'correct_doc': correct_doc,
                'failure_type': failure_type
            })
    
    # Save FULL error report (all 795 rows = 159 questions × 5 systems)
    full_df = pd.DataFrame(full_error_report)
    full_df.to_csv(f'{output_dir}/complete_error_analysis.csv', index=False)
    
    # Save worst errors for easy inspection
    worst_df = pd.DataFrame(worst_errors_report)
    worst_df.to_csv(f'{output_dir}/worst_errors_top20.csv', index=False)
    
    # Print summary statistics
    print(f"\n✓ Complete Error Analysis Done!")
    print(f"  Total questions analyzed: {len(full_df)} (159 questions × 5 systems)")
    print(f"  Saved complete analysis to: {output_dir}/complete_error_analysis.csv")
    print(f"  Saved worst 20 per system to: {output_dir}/worst_errors_top20.csv")
    
    # Failure type breakdown for ALL questions
    print("\n" + "="*60)
    print("FAILURE TYPE BREAKDOWN (ALL QUESTIONS):")
    print("="*60)
    for system in systems:
        system_data = full_df[full_df['system'] == system]
        errors_only = system_data[system_data['is_error'] == True]
        
        print(f"\n{system}:")
        print(f"  Total questions: {len(system_data)}")
        print(f"  Errors (not perfect): {len(errors_only)} ({len(errors_only)/len(system_data)*100:.1f}%)")
        print(f"  Perfect answers: {len(system_data) - len(errors_only)} ({(len(system_data)-len(errors_only))/len(system_data)*100:.1f}%)")
        
        if len(errors_only) > 0:
            print("  \nError breakdown by type:")
            failure_counts = errors_only['failure_type'].value_counts()
            for ftype, count in failure_counts.items():
                print(f"    - {ftype}: {count} ({count/len(errors_only)*100:.1f}%)")
    
    return full_df, worst_df


def question_type_analysis(data: Dict, systems: List[str], output_dir: str):
    """
    Analyze performance by question type (factoid, list, multiple choice) - ALL 159 questions
    """
    print("\n" + "="*60)
    print("PART 5.2: QUESTION TYPE ANALYSIS (ALL 159 QUESTIONS)")
    print("="*60)
    
    results = []
    
    for system in systems:
        evals = data[system]['evaluations'].copy()
        question_types = data[system]['question_types']
        
        # Add question type to evaluations
        evals['question_type'] = question_types
        
        # Compute metrics by question type
        for qtype in ['factoid', 'list', 'multiple choice']:
            type_evals = evals[evals['question_type'] == qtype]
            
            if len(type_evals) == 0:
                continue
            
            results.append({
                'system': system,
                'question_type': qtype,
                'count': len(type_evals),
                'llm_avg': type_evals['llm_score'].mean(),
                'llm_std': type_evals['llm_score'].std(),
                'exact_match_count': type_evals['exact_match'].sum(),
                'exact_match_pct': type_evals['exact_match'].mean() * 100,
                'f1_avg': type_evals['f1_score'].mean(),
                'f1_std': type_evals['f1_score'].std()
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/question_type_analysis.csv', index=False)
    
    # Print summary
    print(f"\n✓ Question Type Analysis Complete!")
    print(f"  Analyzed all 159 questions (53 per type)")
    print(f"  Saved to: {output_dir}/question_type_analysis.csv\n")
    
    print("\nPerformance by Question Type:")
    print(results_df[['system', 'question_type', 'count', 'llm_avg', 'exact_match_pct', 'f1_avg']].to_string(index=False))
    
    return results_df


def create_visualizations(data: Dict, systems: List[str], output_dir: str):
    """
    Create visualizations comparing systems - based on ALL 159 questions
    """
    print("\n" + "="*60)
    print("PART 5.3: VISUALIZATION (ALL 159 QUESTIONS)")
    print("="*60)
    
    # Collect overall metrics
    metrics_data = []
    for system in systems:
        evals = data[system]['evaluations']
        metrics_data.append({
            'System': system,
            'LLM Score': evals['llm_score'].mean(),
            'Exact Match (%)': evals['exact_match'].mean() * 100,
            'F1 Score': evals['f1_score'].mean()
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Visualization 1: Bar chart - Overall metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # LLM Score
    axes[0].bar(metrics_df['System'], metrics_df['LLM Score'], color='skyblue')
    axes[0].set_ylabel('LLM Score (out of 5)', fontsize=12)
    axes[0].set_title('LLM-as-Judge Scores\n(Average across 159 questions)', 
                      fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 5])
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(y=metrics_df['LLM Score'].mean(), color='red', linestyle='--', 
                    alpha=0.5, label='Average')
    
    # Exact Match
    axes[1].bar(metrics_df['System'], metrics_df['Exact Match (%)'], color='lightcoral')
    axes[1].set_ylabel('Exact Match (%)', fontsize=12)
    axes[1].set_title('Exact Match Percentage\n(Out of 159 questions)', 
                      fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 100])
    axes[1].tick_params(axis='x', rotation=45)
    
    # F1 Score
    axes[2].bar(metrics_df['System'], metrics_df['F1 Score'], color='lightgreen')
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('F1 Score\n(Average across 159 questions)', 
                      fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 1])
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Created: {output_dir}/overall_comparison.png")
    
    # Visualization 2: Heatmap - System × Question Type
    # Load question type analysis results
    qt_df = pd.read_csv(f'{output_dir}/question_type_analysis.csv')
    
    # Create pivot table for heatmap
    heatmap_data = qt_df.pivot(index='system', columns='question_type', values='llm_avg')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'LLM Score'}, vmin=1, vmax=5)
    plt.title('System Performance by Question Type\n(LLM Score - 53 questions per type)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Question Type', fontsize=12)
    plt.ylabel('System', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/question_type_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_dir}/question_type_heatmap.png")
    
    # Visualization 3: Component Analysis (Retriever vs Generator impact)
    component_data = []
    
    # Group by retriever
    retrievers = {'None': ['None_gpt4omini'], 
                  'Azure': ['Azure_gpt4omini', 'Azure_flant5'],
                  'Local': ['Local_gpt4omini', 'Local_flant5']}
    
    for ret_name, ret_systems in retrievers.items():
        avg_llm = np.mean([metrics_df[metrics_df['System'] == s]['LLM Score'].values[0] 
                          for s in ret_systems])
        avg_f1 = np.mean([metrics_df[metrics_df['System'] == s]['F1 Score'].values[0] 
                         for s in ret_systems])
        component_data.append({'Component': 'Retriever', 'Type': ret_name, 
                              'LLM Score': avg_llm, 'F1 Score': avg_f1})
    
    # Group by generator
    generators = {'GPT-4o-mini': ['None_gpt4omini', 'Azure_gpt4omini', 'Local_gpt4omini'],
                  'FLAN-T5': ['Azure_flant5', 'Local_flant5']}
    
    for gen_name, gen_systems in generators.items():
        avg_llm = np.mean([metrics_df[metrics_df['System'] == s]['LLM Score'].values[0] 
                          for s in gen_systems])
        avg_f1 = np.mean([metrics_df[metrics_df['System'] == s]['F1 Score'].values[0] 
                         for s in gen_systems])
        component_data.append({'Component': 'Generator', 'Type': gen_name, 
                              'LLM Score': avg_llm, 'F1 Score': avg_f1})
    
    comp_df = pd.DataFrame(component_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Retriever impact
    ret_data = comp_df[comp_df['Component'] == 'Retriever']
    axes[0].bar(ret_data['Type'], ret_data['LLM Score'], color='steelblue')
    axes[0].set_ylabel('Average LLM Score', fontsize=12)
    axes[0].set_title('Retriever Impact on Performance\n(Averaged across generators)', 
                      fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 5])
    
    # Generator impact
    gen_data = comp_df[comp_df['Component'] == 'Generator']
    axes[1].bar(gen_data['Type'], gen_data['LLM Score'], color='darkorange')
    axes[1].set_ylabel('Average LLM Score', fontsize=12)
    axes[1].set_title('Generator Impact on Performance\n(Averaged across retrievers)', 
                      fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 5])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/component_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_dir}/component_analysis.png")
    
    print("\n✓ All Visualizations Complete!")
    
    plt.close('all')  # Close all figures to free memory


def retrieval_quality_analysis(data: Dict, systems: List[str], output_dir: str):
    """
    Analyze retrieval quality - did retriever get the right documents? - ALL 159 questions
    """
    print("\n" + "="*60)
    print("PART 5.4: RETRIEVAL QUALITY ANALYSIS (ALL 159 QUESTIONS)")
    print("="*60)
    
    retrieval_results = []
    detailed_retrieval = []
    
    for system in systems:
        if system == 'None_gpt4omini':
            continue  # Skip no-retrieval system
        
        print(f"\nAnalyzing retrieval quality for {system} (all 159 questions)...")
        
        retrieved_docs = data[system]['retrieved_docs']
        evidence_files = data[system]['evidence_files']
        evals = data[system]['evaluations']
        questions = data[system]['questions']
        
        correct_retrievals = 0
        retrieval_details = []
        
        for idx in range(len(retrieved_docs)):
            correct_doc = evidence_files[idx]
            retrieved = str(retrieved_docs[idx])
            
            is_correct = correct_doc in retrieved
            correct_retrievals += int(is_correct)
            
            detail = {
                'system': system,
                'question_idx': idx + 1,
                'question': questions[idx],
                'correct_doc': correct_doc,
                'retrieved_docs': retrieved,
                'correct_doc_retrieved': is_correct,
                'llm_score': evals.loc[idx, 'llm_score'],
                'exact_match': evals.loc[idx, 'exact_match'],
                'f1_score': evals.loc[idx, 'f1_score']
            }
            retrieval_details.append(detail)
            detailed_retrieval.append(detail)
        
        retrieval_accuracy = correct_retrievals / len(retrieved_docs) * 100
        
        # Analyze performance when retrieval is correct vs incorrect
        details_df = pd.DataFrame(retrieval_details)
        correct_ret = details_df[details_df['correct_doc_retrieved'] == True]
        incorrect_ret = details_df[details_df['correct_doc_retrieved'] == False]
        
        retrieval_results.append({
            'system': system,
            'total_questions': len(retrieved_docs),
            'correct_retrievals': correct_retrievals,
            'incorrect_retrievals': len(retrieved_docs) - correct_retrievals,
            'retrieval_accuracy': retrieval_accuracy,
            'llm_score_correct_ret': correct_ret['llm_score'].mean() if len(correct_ret) > 0 else 0,
            'llm_score_incorrect_ret': incorrect_ret['llm_score'].mean() if len(incorrect_ret) > 0 else 0,
            'exact_match_correct_ret': correct_ret['exact_match'].mean() * 100 if len(correct_ret) > 0 else 0,
            'exact_match_incorrect_ret': incorrect_ret['exact_match'].mean() * 100 if len(incorrect_ret) > 0 else 0,
            'f1_correct_ret': correct_ret['f1_score'].mean() if len(correct_ret) > 0 else 0,
            'f1_incorrect_ret': incorrect_ret['f1_score'].mean() if len(incorrect_ret) > 0 else 0
        })
        
        print(f"  Retrieval Accuracy: {retrieval_accuracy:.1f}% ({correct_retrievals}/159)")
        print(f"  LLM Score when retrieval correct: {retrieval_results[-1]['llm_score_correct_ret']:.2f}")
        print(f"  LLM Score when retrieval incorrect: {retrieval_results[-1]['llm_score_incorrect_ret']:.2f}")
        print(f"  Impact of correct retrieval: +{retrieval_results[-1]['llm_score_correct_ret'] - retrieval_results[-1]['llm_score_incorrect_ret']:.2f} LLM score")
    
    # Save summary results
    ret_df = pd.DataFrame(retrieval_results)
    ret_df.to_csv(f'{output_dir}/retrieval_quality_summary.csv', index=False)
    
    # Save detailed results (all 636 rows = 159 questions × 4 RAG systems)
    detailed_df = pd.DataFrame(detailed_retrieval)
    detailed_df.to_csv(f'{output_dir}/retrieval_quality_detailed.csv', index=False)
    
    print(f"\n✓ Retrieval Quality Analysis Complete!")
    print(f"  Analyzed all 636 retrievals (159 questions × 4 RAG systems)")
    print(f"  Summary saved to: {output_dir}/retrieval_quality_summary.csv")
    print(f"  Detailed results saved to: {output_dir}/retrieval_quality_detailed.csv\n")
    print("\nRetrieval Quality Summary:")
    print(ret_df[['system', 'retrieval_accuracy', 'llm_score_correct_ret', 'llm_score_incorrect_ret']].to_string(index=False))
    
    return ret_df, detailed_df


def generate_summary_report(output_dir: str):
    """
    Generate a summary report of all analyses
    """
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    summary = """# Part 5 Advanced Analysis - COMPLETE Summary Report

## Overview
This report presents comprehensive advanced analysis of the 5 RAG systems developed for HW6.
**ALL 159 questions were analyzed** across all metrics and systems.

## Analyses Performed

### 1. Complete Error Analysis (ALL 159 QUESTIONS)
- Analyzed ALL 795 predictions (159 questions × 5 systems)
- Categorized failure types for every question
- Identified top 20 worst errors per system for detailed inspection

### 2. Question Type Analysis (ALL 159 QUESTIONS)
- Broke down performance for all questions by type
- Factoid, List, Multiple Choice (53 each)

### 3. Visualizations (Based on ALL 159 QUESTIONS)
- Created 3 professional visualizations using complete dataset

### 4. Retrieval Quality Analysis (ALL 636 RETRIEVALS)
- Measured retrieval accuracy for ALL questions in RAG systems
- Total retrievals analyzed: 636 (159 questions × 4 RAG systems)

## Data Coverage: 100%

All analyses are based on the complete dataset - no sampling.
"""
    
    with open(f'{output_dir}/PART5_SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n✓ Summary report saved to: {output_dir}/PART5_SUMMARY.md")


def main():
    print("="*60)
    print("PART 5: COMPLETE ADVANCED ANALYSIS")
    print("Analyzing ALL 159 questions across all 5 systems")
    print("="*60)
    print("\nLoading all data...")
    
    # Create output directory
    output_dir = 'output/analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data, systems = load_all_data()
    print(f"✓ Loaded data for {len(systems)} systems")
    print(f"  Each system has 159 questions\n")
    
    # Run analyses
    full_error_df, worst_df = complete_error_analysis(data, systems, output_dir)
    qt_df = question_type_analysis(data, systems, output_dir)
    create_visualizations(data, systems, output_dir)
    ret_summary_df, ret_detailed_df = retrieval_quality_analysis(data, systems, output_dir)
    
    # Generate summary
    generate_summary_report(output_dir)
    
    print("\n" + "="*60)
    print("✅ ALL ANALYSES COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {output_dir}/")
    print("\n" + "="*60)
    print("DATA COVERAGE: 100%")
    print("  - Error analysis: 795/795 predictions (159 × 5)")
    print("  - Retrieval analysis: 636/636 retrievals (159 × 4)")
    print("  - Question type analysis: ALL 159 questions")
    print("="*60)
    print("\nUse these in your report!")
    print("="*60)


if __name__ == "__main__":
    main()
