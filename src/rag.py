"""
RAG Pipeline - Main script for running QA systems
Usage: python rag.py --retriever <azure|local|none> --generator <gpt4omini|flant5>
"""

import argparse
import pandas as pd
import os
from tqdm import tqdm
from retriever import get_retriever
from generator import get_generator


def load_questions(filepath: str):
    """Load questions from TSV file"""
    df = pd.read_csv(filepath, sep='\t', header=None, names=['question', 'type'])
    return df


def run_rag_system(questions_df, retriever, generator, retriever_name, generator_name, output_dir):
    """Run complete RAG system and save predictions"""
    
    print(f"\n{'='*60}")
    print(f"Running System: {retriever_name}_{generator_name}")
    print(f"{'='*60}\n")
    
    predictions = []
    additional_info = []
    
    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing questions"):
        question = row['question']
        
        # Step 1: Retrieve documents (if retriever exists)
        if retriever:
            retrieved_docs = retriever.retrieve(question, top_k=3)
            # Combine retrieved documents as context
            context = "\n\n".join([doc[0] for doc in retrieved_docs])
            doc_names = ", ".join([doc[1] for doc in retrieved_docs])
        else:
            context = None
            doc_names = "None"
        
        # Step 2: Generate answer
        answer = generator.generate(question, context)
        
        predictions.append(answer)
        additional_info.append(doc_names)
    
    # Save predictions
    output_filename = f"{retriever_name}_{generator_name}.tsv"
    output_path = os.path.join(output_dir, output_filename)
    
    output_df = pd.DataFrame({
        'prediction': predictions,
        'retrieved_docs': additional_info
    })
    
    output_df.to_csv(output_path, sep='\t', header=False, index=False)
    
    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"  Total questions processed: {len(predictions)}\n")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run RAG QA System")
    parser.add_argument('--retriever', type=str, required=True,
                        choices=['azure', 'local', 'none'],
                        help='Retriever type: azure, local, or none')
    parser.add_argument('--generator', type=str, required=True,
                        choices=['gpt4omini', 'flant5'],
                        help='Generator type: gpt4omini or flant5')
    parser.add_argument('--questions', type=str, default='data/question.tsv',
                        help='Path to questions TSV file')
    parser.add_argument('--corpus', type=str, default='data/corpus',
                        help='Path to corpus directory')
    parser.add_argument('--output', type=str, default='output/prediction',
                        help='Output directory for predictions')
    parser.add_argument('--api-key', type=str, default=None,
                        help='CMU AI Gateway API key (for API-based models)')
    parser.add_argument('--base-url', type=str, 
                        default='https://ai-gateway.andrew.cmu.edu/',
                        help='API base URL')
    
    args = parser.parse_args()
    
    # Load questions
    print(f"Loading questions from: {args.questions}")
    questions_df = load_questions(args.questions)
    print(f"✓ Loaded {len(questions_df)} questions\n")
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.getenv('CMU_API_KEY')
    
    if not api_key and (args.retriever == 'azure' or args.generator == 'gpt4omini'):
        raise ValueError("API key required! Set CMU_API_KEY environment variable or use --api-key")
    
    # Initialize retriever
    print(f"Initializing retriever: {args.retriever}")
    retriever = get_retriever(
        args.retriever,
        args.corpus,
        api_key=api_key,
        base_url=args.base_url
    )
    
    # Initialize generator
    print(f"\nInitializing generator: {args.generator}")
    generator = get_generator(
        args.generator,
        api_key=api_key,
        base_url=args.base_url
    )
    
    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)
    
    # Determine retriever name for output file
    retriever_name = "None" if args.retriever == "none" else args.retriever.capitalize()
    generator_name = args.generator
    
    # Run system
    run_rag_system(
        questions_df,
        retriever,
        generator,
        retriever_name,
        generator_name,
        args.output
    )
    
    print("="*60)
    print("✅ DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
