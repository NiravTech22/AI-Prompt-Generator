#!/usr/bin/env python3
"""
AI Prompt Generator - Python Backend
Generates curated AI prompts from 4-word descriptions
Can be used as a standalone script or integrated with the browser extension
"""

import json
import random
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    template: str
    category: str
    complexity: str  # 'simple', 'medium', 'advanced'

class AIPromptGenerator:
    def __init__(self):
        self.prompt_templates = {
            'creative': [
                PromptTemplate(
                    "Create a {topic} that is {style} and {audience}-focused, incorporating {element} to maximize engagement and impact.",
                    'creative', 'medium'
                ),
                PromptTemplate(
                    "Design an innovative {topic} strategy that combines {style} approach with {audience} targeting, emphasizing {element}.",
                    'creative', 'advanced'
                ),
                PromptTemplate(
                    "Develop a compelling {topic} concept that leverages {style} techniques to connect with {audience} through {element}.",
                    'creative', 'medium'
                ),
                PromptTemplate(
                    "Generate a {style} {topic} that resonates with {audience} by highlighting {element} in an authentic way.",
                    'creative', 'simple'
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

    def generate_prompt(self, words: List[str], complexity: str = 'medium') -> Dict[str, str]:
        """Generate a curated prompt from 4 words."""
        if len(words) != 4:
            raise ValueError("Exactly 4 words are required")
        
        # Clean and validate words
        words = [word.strip() for word in words if word.strip()]
        if len(words) != 4:
            raise ValueError("All 4 words must be non-empty")
        
        # Detect category and get appropriate templates
        category = self.detect_category(words)
        templates = [t for t in self.prompt_templates[category] 
                   if t.complexity == complexity or complexity == 'any']
        
        if not templates:
            templates = self.prompt_templates[category]
        
        # Select random template
        selected_template = random.choice(templates)
        
        # Assign words to slots intelligently
        assignments = self.assign_words_to_slots(words)
        
        # Generate the prompt
        prompt = selected_template.template.format(
            topic=assignments['topic'],
            style=assignments['style'],
            audience=assignments['audience'],
            element=assignments['element']
        )
        
        return {
            'prompt': prompt,
            'category': category,
            'complexity': selected_template.complexity,
            'words_used': words,
            'assignments': assignments
        }

    def generate_multiple_prompts(self, words: List[str], count: int = 3) -> List[Dict[str, str]]:
        """Generate multiple prompt variations."""
        prompts = []
        complexities = ['simple', 'medium', 'advanced']
        
        for i in range(count):
            complexity = complexities[i % len(complexities)]
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