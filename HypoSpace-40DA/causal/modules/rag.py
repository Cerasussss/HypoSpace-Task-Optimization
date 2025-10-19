"""
Retrieval Augmented Generation (RAG) module for causal graph discovery.
"""
import json
import numpy as np
from typing import List, Dict
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CausalRAG:
    """Retrieval Augmented Generation for causal graph discovery."""

    def __init__(self, examples_file: str = "few_shot_examples.json"):
        """
        Initialize RAG with few-shot examples.

        Args:
            examples_file: Path to JSON file containing few-shot examples
        """
        self.examples_file = examples_file
        if not Path(examples_file).exists():
            raise FileNotFoundError(f"Examples file not found: {examples_file}")

        with open(examples_file, 'r') as f:
            self.examples_data = json.load(f)

        self.examples = self.examples_data['examples']
        self.vectorizer = TfidfVectorizer()
        self._build_index()

    def _build_index(self):
        """Build TF-IDF index for examples."""
        # Create text representations of inputs for indexing
        texts = []
        for example in self.examples:
            obs_strings = [obs['string'] for obs in example['input']['observations']]
            text = '; '.join(obs_strings)
            texts.append(text)

        # Fit vectorizer and transform texts
        self.example_vectors = self.vectorizer.fit_transform(texts)

    def retrieve_similar_examples(self, observations: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k similar examples based on observation similarity.

        Args:
            observations: List of observation dictionaries
            top_k: Number of examples to retrieve

        Returns:
            List of similar examples with their hypotheses
        """
        # Create text representation of query observations
        obs_strings = [obs['string'] for obs in observations]
        query_text = '; '.join(obs_strings)

        # Vectorize query
        query_vector = self.vectorizer.transform([query_text])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.example_vectors).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return examples with similarities above threshold
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append({
                    'example': self.examples[idx],
                    'similarity': float(similarities[idx])
                })

        return results

    def format_examples_for_prompt(self, examples: List[Dict]) -> str:
        """
        Format retrieved examples for inclusion in LLM prompt.

        Args:
            examples: List of examples with similarity scores

        Returns:
            Formatted string for prompt inclusion
        """
        if not examples:
            return ""

        formatted = "\n\n[Reference Examples]\n"
        for i, item in enumerate(examples):
            example = item['example']
            formatted += f"Example {i + 1} (similarity: {item['similarity']:.2f}):\n"
            formatted += f"  Input: {'; '.join([obs['string'] for obs in example['input']['observations']])}\n"
            formatted += "  Possible hypotheses:\n"
            for j, hyp in enumerate(example['hypotheses'][:3]):  # Limit to top 3 for brevity
                formatted += f"    {j + 1}. {hyp['hypothesis']}\n"
                formatted += f"       Explanation: {hyp['explanation'][:100]}...\n"
            formatted += "\n"

        return formatted


# Test the RAG module
if __name__ == "__main__":
    # Test with a sample observation
    try:
        rag = CausalRAG()

        test_observations = [
            {
                "perturbed_node": "A",
                "effects": {
                    "A": 0,
                    "B": 1,
                    "C": 1
                },
                "string": "Perturb(A) -> A:0 B:1 C:1"
            }
        ]

        similar_examples = rag.retrieve_similar_examples(test_observations, top_k=2)
        print("Retrieved examples:")
        for example in similar_examples:
            print(f"Similarity: {example['similarity']:.2f}")
            print(f"Example: {example['example']['id']}")
    except Exception as e:
        print(f"Error testing RAG module: {e}")