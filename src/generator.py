"""
Generator Module - Handles answer generation using API or open-weight models
"""

from typing import List, Tuple
from openai import OpenAI


class BaseGenerator:
    """Base class for all generators"""
    
    def generate(self, question: str, context: str = None) -> str:
        """Generate answer for question, optionally using context"""
        raise NotImplementedError


class GPT4oMiniGenerator(BaseGenerator):
    """API-based generator using GPT-4o-mini"""
    
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "gpt-4o-mini-2024-07-18"
        print(f"✓ GPT-4o-mini generator ready")
    
    def generate(self, question: str, context: str = None) -> str:
        """Generate answer using GPT-4o-mini"""
        
        if context:
            # RAG mode: use context
            system_prompt = """You are an expert on Basketball in Africa. 
Answer questions concisely and accurately based on the provided context.
If the context doesn't contain enough information, say so briefly."""
            
            user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        else:
            # No-retrieval mode
            system_prompt = "You are an expert on Basketball in Africa. Answer questions concisely and accurately."
            user_prompt = question
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"ERROR: {str(e)}"


class FlanT5Generator(BaseGenerator):
    """Open-weight generator using FLAN-T5-base"""
    
    def __init__(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            
            print("Loading FLAN-T5-base model (this may take a minute)...")
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            
            # Use GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            print(f"✓ FLAN-T5-base generator ready (device: {self.device})")
        
        except ImportError:
            raise ImportError("Please install: pip install transformers torch")
    
    def generate(self, question: str, context: str = None) -> str:
        """Generate answer using FLAN-T5"""
        
        if context:
            # RAG mode: construct prompt with context
            # FLAN-T5 works well with explicit instructions
            prompt = f"""Answer the question based on the context below. Be concise.

Context: {context[:2000]}

Question: {question}

Answer:"""
        else:
            # No-retrieval mode
            prompt = f"Answer this question about Basketball in Africa: {question}"
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                early_stopping=True,
                temperature=0.7
            )
            
            # Decode
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer.strip()
        
        except Exception as e:
            return f"ERROR: {str(e)}"


def get_generator(generator_type: str, api_key: str = None, base_url: str = None):
    """Factory function to get appropriate generator"""
    
    if generator_type.lower() == "gpt4omini":
        if not api_key or not base_url:
            raise ValueError("API key and base URL required for GPT-4o-mini")
        return GPT4oMiniGenerator(api_key, base_url)
    
    elif generator_type.lower() == "flant5":
        return FlanT5Generator()
    
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
