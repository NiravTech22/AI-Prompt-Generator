#!/usr/bin/env python3
"""
AI Prompt Generator - Python Backend
Generates curated AI prompts from 4-word descriptions
Can be used as a standalone script or integrated with the browser extension
"""

import json
import random
import re
import os
import base64
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
# Add spaCy for advanced NLP
try:
    import spacy  # type: ignore
    nlp = spacy.load('en_core_web_sm')
except Exception:
    nlp = None
# NOTE: Requires: pip install torch transformers
import openai  # type: ignore

PROMPT_HISTORY_FILE = 'prompts_history.json'
ATTEMPT_FILE = 'user_attempts.txt'

# Set your OpenAI API key (for production, use environment variables!)
openai.api_key = "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"

def encrypt(text: str) -> str:
    return base64.b64encode(text.encode('utf-8')).decode('utf-8')

def decrypt(text: str) -> str:
    try:
        return base64.b64decode(text.encode('utf-8')).decode('utf-8')
    except Exception:
        return ''

def save_prompt_to_history(prompt: str):
    history = []
    if os.path.exists(PROMPT_HISTORY_FILE):
        with open(PROMPT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                history = json.load(f)
            except Exception:
                history = []
    history.append({'prompt': encrypt(prompt)})
    with open(PROMPT_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f)

def get_prompt_history() -> list:
    if not os.path.exists(PROMPT_HISTORY_FILE):
        return []
    with open(PROMPT_HISTORY_FILE, 'r', encoding='utf-8') as f:
        try:
            history = json.load(f)
        except Exception:
            return []
    return [decrypt(item['prompt']) for item in history if 'prompt' in item]

def get_attempt_count() -> int:
    if not os.path.exists(ATTEMPT_FILE):
        return 0
    try:
        with open(ATTEMPT_FILE, 'r') as f:
            return int(f.read().strip())
    except Exception:
        return 0

def increment_attempt_count():
    count = get_attempt_count() + 1
    with open(ATTEMPT_FILE, 'w') as f:
        f.write(str(count))
    return count

def reset_attempt_count():
    if os.path.exists(ATTEMPT_FILE):
        os.remove(ATTEMPT_FILE)

@dataclass
class PromptTemplate:
    template: str
    category: str
    complexity: str  # 'simple', 'medium', 'advanced'
    keywords: list = field(default_factory=list)  # Default to empty list

class AIPromptGenerator:
    def __init__(self):
        self.prompt_templates = {
            'creative': [
                PromptTemplate(
                    "Create a {topic} that is {style} and {audience}-focused, incorporating {element} to maximize engagement and impact.",
                    'creative', 'medium', ['create', 'engagement', 'impact', 'creative']
                ),
                PromptTemplate(
                    "Design an innovative {topic} strategy that combines {style} approach with {audience} targeting, emphasizing {element}.",
                    'creative', 'advanced', ['design', 'innovative', 'strategy', 'creative']
                ),
                PromptTemplate(
                    "Develop a compelling {topic} concept that leverages {style} techniques to connect with {audience} through {element}.",
                    'creative', 'medium', ['develop', 'compelling', 'concept', 'creative']
                ),
                PromptTemplate(
                    "Generate a {style} {topic} that resonates with {audience} by highlighting {element} in an authentic way.",
                    'creative', 'simple', ['generate', 'resonate', 'highlight', 'creative']
                )
            ],
            'technical': [
                PromptTemplate(
                    "Build a {topic} solution using {style} methodology that addresses {audience} needs, implementing {element} for optimal results.",
                    'technical', 'advanced'
                ),
                PromptTemplate(
                    "Architect a {topic} system with {style} design patterns, optimized for {audience} requirements and {element} integration.",
                    'technical', 'advanced'
                ),
                PromptTemplate(
                    "Create a {style} {topic} implementation for {audience} that focuses on {element} functionality.",
                    'technical', 'medium'
                ),
                PromptTemplate(
                    "Develop a {topic} using {style} approach, designed for {audience} with {element} as core feature.",
                    'technical', 'simple'
                )
            ],
            'business': [
                PromptTemplate(
                    "Develop a {topic} strategy that utilizes {style} approach to engage {audience}, focusing on {element} for competitive advantage.",
                    'business', 'advanced'
                ),
                PromptTemplate(
                    "Create a comprehensive {topic} plan with {style} methodology, targeting {audience} while emphasizing {element} outcomes.",
                    'business', 'medium'
                ),
                PromptTemplate(
                    "Design a {topic} initiative using {style} framework to reach {audience}, with {element} as the key differentiator.",
                    'business', 'medium'
                ),
                PromptTemplate(
                    "Build a {style} {topic} approach for {audience} that prioritizes {element} results.",
                    'business', 'simple'
                )
            ],
            'content': [
                PromptTemplate(
                    "Craft {topic} content with {style} tone that resonates with {audience}, highlighting {element} throughout the narrative.",
                    'content', 'medium'
                ),
                PromptTemplate(
                    "Produce engaging {topic} material using {style} storytelling techniques for {audience}, centered around {element}.",
                    'content', 'advanced'
                ),
                PromptTemplate(
                    "Write compelling {topic} copy with {style} voice that speaks to {audience}, featuring {element} prominently.",
                    'content', 'medium'
                ),
                PromptTemplate(
                    "Create {style} {topic} content for {audience} that emphasizes {element}.",
                    'content', 'simple'
                )
            ],
            'analysis': [
                PromptTemplate(
                    "Analyze {topic} using {style} methodology to provide insights for {audience}, with focus on {element} implications.",
                    'analysis', 'advanced'
                ),
                PromptTemplate(
                    "Examine {topic} through {style} lens to deliver actionable recommendations for {audience}, emphasizing {element}.",
                    'analysis', 'medium'
                ),
                PromptTemplate(
                    "Research {topic} with {style} approach to generate insights for {audience} about {element}.",
                    'analysis', 'medium'
                ),
                PromptTemplate(
                    "Study {topic} using {style} methods for {audience}, focusing on {element}.",
                    'analysis', 'simple'
                )
            ]
        }
        
        self.category_keywords = {
            'creative': ['creative', 'design', 'art', 'visual', 'brand', 'marketing', 'campaign', 'content', 'story', 'narrative', 'logo', 'poster', 'video'],
            'technical': ['code', 'software', 'app', 'system', 'database', 'api', 'algorithm', 'program', 'development', 'tech', 'website', 'mobile', 'cloud'],
            'business': ['strategy', 'business', 'sales', 'revenue', 'market', 'growth', 'plan', 'roi', 'profit', 'customer', 'startup', 'finance', 'management'],
            'content': ['blog', 'article', 'post', 'copy', 'write', 'content', 'social', 'email', 'newsletter', 'script', 'book', 'tutorial', 'guide'],
            'analysis': ['analyze', 'research', 'data', 'report', 'study', 'insights', 'metrics', 'trends', 'evaluation', 'assessment', 'survey', 'statistics']
        }
        
        # Word classification for smarter assignment
        self.word_types = {
            'style': ['modern', 'creative', 'professional', 'casual', 'formal', 'innovative', 'traditional', 'minimalist', 'bold', 'elegant'],
            'audience': ['customers', 'users', 'clients', 'students', 'professionals', 'team', 'audience', 'visitors', 'subscribers', 'members'],
            'action': ['create', 'build', 'design', 'develop', 'generate', 'write', 'analyze', 'research', 'optimize', 'improve'],
            'element': ['features', 'benefits', 'value', 'experience', 'interface', 'content', 'strategy', 'approach', 'method', 'solution']
        }

    def detect_category(self, words: List[str]) -> str:
        """Detect the most appropriate category based on input words."""
        word_string = ' '.join(words).lower()
        max_score = 0
        best_category = 'creative'  # Default category
        
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in word_string)
            if score > max_score:
                max_score = score
                best_category = category
        
        return best_category

    def classify_word(self, word: str) -> str:
        """Classify a word into its most likely type (style, audience, etc.)."""
        word_lower = word.lower()
        
        for word_type, type_words in self.word_types.items():
            if any(type_word in word_lower for type_word in type_words):
                return word_type
        
        return 'general'  # Default classification

    def assign_words_to_slots(self, words: List[str]) -> Dict[str, str]:
        """Intelligently assign words to prompt template slots."""
        assignments = {
            'topic': words[0],
            'style': words[1], 
            'audience': words[2],
            'element': words[3]
        }
        
        # Classify each word and reassign if better fit found
        word_classifications = [(word, self.classify_word(word)) for word in words]
        
        # Reassign based on classifications
        for word, classification in word_classifications:
            if classification in assignments:
                # Move current assignment to a free slot
                current_value = assignments[classification]
                assignments[classification] = word
                
                # Find a new slot for the displaced word
                for slot in assignments:
                    if assignments[slot] == word and slot != classification:
                        assignments[slot] = current_value
                        break
        
        return assignments

    def nlp_assign_words_to_slots(self, words: List[str]) -> Dict[str, str]:
        """Use spaCy NLP to assign words to slots more intelligently."""
        if not nlp:
            return self.assign_words_to_slots(words)
        doc = nlp(' '.join(words))
        assignments = {'topic': '', 'style': '', 'audience': '', 'element': ''}
        used = set()
        # Try to use NER and POS to assign
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART'] and not assignments['topic']:
                assignments['topic'] = str(ent.text)
                used.add(ent.text)
            elif ent.label_ in ['PERSON', 'NORP', 'GROUP'] and not assignments['audience']:
                assignments['audience'] = str(ent.text)
                used.add(ent.text)
        for token in doc:
            if token.text in used:
                continue
            if token.pos_ == 'ADJ' and not assignments['style']:
                assignments['style'] = str(token.text)
                used.add(token.text)
            elif token.pos_ == 'NOUN' and not assignments['element']:
                assignments['element'] = str(token.text)
                used.add(token.text)
        # Fallback for any empty slots
        for i, slot in enumerate(['topic', 'style', 'audience', 'element']):
            if not assignments[slot]:
                assignments[slot] = str(words[i])
        return assignments

    def semantic_score(self, words: List[str], template: PromptTemplate) -> int:
        """Score a template based on overlap with input words and template keywords."""
        if not template.keywords:
            return 0
        score = 0
        for word in words:
            for kw in template.keywords:
                if word.lower() in kw.lower() or kw.lower() in word.lower():
                    score += 1
        return score

    def contains_inappropriate_language(self, words: List[str]) -> bool:
        banned_words = [
            'fuck', 'shit', 'bitch', 'asshole', 'bastard', 'dick', 'cunt', 'nigger', 'fag', 'slut', 'whore', 'rape', 'retard', 'chink', 'spic', 'kike', 'twat', 'wank', 'cock', 'pussy', 'cum', 'faggot', 'nigga', 'motherfucker', 'fucker', 'douche', 'bollocks', 'bugger', 'bollok', 'arse', 'wanker', 'tosser', 'prick', 'dyke', 'tranny', 'coon', 'gook', 'spook', 'tard', 'homo', 'queer', 'kraut', 'gyp', 'gyppo', 'golliwog', 'negro', 'paki', 'raghead', 'sandnigger', 'skank', 'spade', 'wetback', 'zipperhead'
        ]
        for word in words:
            for bw in banned_words:
                if bw in word.lower():
                    return True
        return False

    def generate_structured_prompt(self, words: list, user_history: list = None) -> str:
        """
        Generate a deeply structured prompt using OpenAI GPT-4 API,
        optionally conditioning on user history.
        """
        # Build a context string from user history
        history_context = ""
        if user_history and len(user_history) > 0:
            history_context = "User has previously requested: " + "; ".join(user_history[-3:]) + ". "
        # Build a detailed instruction for the model
        instruction = (
            f"{history_context}Create a detailed, structured prompt for an AI system. "
            f"Context: {words[0]}. Domain: {words[1]}. Format: {words[2]}. Goal: {words[3]}. "
            "The prompt should be clear, actionable, and ready to use in another AI system."
        )
        # Call OpenAI GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert prompt engineer."},
                {"role": "user", "content": instruction}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content'].strip()

    def generate_prompt(self, words: list, complexity: str = None) -> Dict[str, Any]:
        # Only use explicit complexity, default to 'medium'. Do not infer.
        complexity = complexity if complexity and isinstance(complexity, str) else 'medium'
        if self.contains_inappropriate_language(words):
            count = increment_attempt_count()
            if count >= 2:
                raise Exception('Access restricted due to inappropriate language.')
            else:
                raise Exception('Inappropriate language detected. One more attempt will result in restriction.')
        reset_attempt_count()
        if len(words) != 4:
            raise ValueError("Exactly 4 words are required")
        words = [str(word).strip() for word in words if str(word).strip()]
        if len(words) != 4:
            raise ValueError("All 4 words must be non-empty")
        # Use local model for prompt generation
        history = get_prompt_history()
        structured_prompt = self.generate_structured_prompt(words, history)
        save_prompt_to_history(structured_prompt)
        return {
            'prompt': structured_prompt,
            'history': history
        }

    def explain_prompt_choice(self, words: List[str], template: PromptTemplate, assignments: Dict[str, str]) -> str:
        """Explain why this prompt/template was chosen."""
        explanation = f"Selected template for category '{template.category}' and complexity '{template.complexity}'. "
        if template.keywords:
            overlap = [w for w in words if any(w.lower() in k.lower() or k.lower() in w.lower() for k in template.keywords)]
            explanation += f"Matched keywords: {', '.join(overlap)}. "
        explanation += f"Slot assignments: {assignments}."
        return explanation

    def generate_multiple_prompts(self, words: List[str], count: int = 3) -> List[Dict[str, Any]]:
        if self.contains_inappropriate_language(words):
            raise Exception('Access restricted due to inappropriate language.')
        prompts = []
        complexities = ['simple', 'medium', 'advanced']
        for i in range(count):
            complexity = str(complexities[i % len(complexities)])
            try:
                prompt_data = self.generate_prompt(words, complexity)
                prompts.append(prompt_data)
            except Exception as e:
                print(f"Error generating prompt {i+1}: {e}")
        return prompts

def main():
    """Command line interface for the prompt generator."""
    generator = AIPromptGenerator()
    
    print("🤖 AI Prompt Generator")
    print("=" * 50)
    print("Enter 4 words to generate a curated AI prompt")
    print("Example: creative marketing email campaign")
    print()
    
    while True:
        try:
            user_input = input("Enter 4 words (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            words = user_input.split()
            
            if len(words) != 4:
                print(f"❌ Please enter exactly 4 words. You entered {len(words)} words.")
                continue
            
            # Generate single prompt
            result = generator.generate_prompt(words)
            
            print(f"\n✅ Generated Prompt ({result['category']} - {result['complexity']}):")
            print("-" * 60)
            print(f"{result['prompt']}")
            print("-" * 60)
            print(f"Words used: {', '.join(result['words_used'])}")
            print(f"Category: {result['category'].title()}")
            print(f"Complexity: {result['complexity'].title()}")
            print()
            
            # Ask if user wants multiple variations
            want_more = input("Generate more variations? (y/n): ").strip().lower()
            if want_more in ['y', 'yes']:
                variations = generator.generate_multiple_prompts(words, 3)
                print(f"\n🎯 Additional Variations:")
                for i, var in enumerate(variations, 1):
                    print(f"\n{i}. {var['prompt']}")
            
            print("\n" + "="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nsee ya!")
            break
        except Exception as e:
            print(f"something went wrong: {e}")
            print("please try again with 4 valid words.\n")

def test_generator():
    """test function to verify the generator works correctly."""
    generator = AIPromptGenerator()
    
    test_cases = [
        ["creative", "marketing", "email", "campaign"],
        ["modern", "website", "design", "portfolio"],
        ["data", "analysis", "sales", "report"],
        ["mobile", "app", "fitness", "tracking"],
        ["social", "media", "content", "strategy"]
    ]
    
    print("Testing AI Prompt Generator")
    print("=" * 50)
    
    for i, words in enumerate(test_cases, 1):
        try:
            result = generator.generate_prompt(words)
            print(f"\nTest {i}: {' '.join(words)}")
            print(f"Category: {result['category']}")
            print(f"Prompt: {result['prompt']}")
            print("-" * 30)
        except Exception as e:
            print(f"Test {i} failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_generator()
        elif len(sys.argv) == 5:
            # Command line usage: python prompt_generator.py word1 word2 word3 word4
            generator = AIPromptGenerator()
            words = sys.argv[1:5]
            try:
                result = generator.generate_prompt(words)
                print(result['prompt'])
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Usage:")
            print("  python prompt_generator.py                    # Interactive mode")
            print("  python prompt_generator.py test               # Run tests")
            print("  python prompt_generator.py word1 word2 word3 word4  # Generate single prompt")
    else:
        main()