"""
Retrieval Augmented Generation (RAG) module for causal graph discovery.
"""
import json
import numpy as np
from typing import List, Dict
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


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
        # Use a more appropriate vectorizer for our use case
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            max_features=1000,  # Limit vocabulary size
            token_pattern=r'\b\w+\b'  # Token pattern
        )
        self._build_index()

    def _build_index(self):
        """Build TF-IDF index for examples."""
        # Create text representations of inputs for indexing
        texts = []
        for example in self.examples:
            obs_strings = [obs['string'] for obs in example['input']['observations']]
            # Create a more structured representation
            text = ' | '.join(obs_strings)  # Use pipe to separate observations
            texts.append(text)

        # Fit vectorizer and transform texts
        self.example_vectors = self.vectorizer.fit_transform(texts)
        # Safely get shape information
        shape_info = getattr(self.example_vectors, 'shape', 'unknown')
        if hasattr(shape_info, '__getitem__'):
            features_count = shape_info[1]
        else:
            features_count = 'unknown'
        print(f"Built index with {len(texts)} examples and {features_count} features")

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
        query_text = ' | '.join(obs_strings)  # Use same format as index

        # Vectorize query
        query_vector = self.vectorizer.transform([query_text])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.example_vectors).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return examples with similarities above threshold
        results = []
        for idx in top_indices:
            # Lower the threshold to allow more examples to be returned
            if similarities[idx] > 0.01:  # Minimum similarity threshold
                results.append({
                    'example': self.examples[idx],
                    'similarity': float(similarities[idx])
                })

        return results

    def retrieve_diverse_examples(self, observations: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k diverse examples based on observation similarity and diversity.

        Args:
            observations: List of observation dictionaries
            top_k: Number of examples to retrieve

        Returns:
            List of diverse examples with their hypotheses
        """
        # First retrieve more candidates
        candidate_k = min(top_k * 3, len(self.examples))
        candidates = self.retrieve_similar_examples(observations, candidate_k)

        if len(candidates) <= top_k:
            return candidates

        # Extract vectors for candidates
        candidate_indices = []
        for candidate in candidates:
            # Find index of this example
            for i, example in enumerate(self.examples):
                if example['id'] == candidate['example']['id']:
                    candidate_indices.append(i)
                    break

        # Get vectors for candidates using proper indexing
        from scipy.sparse import vstack
        candidate_vectors_list = [self.example_vectors[i] for i in candidate_indices]
        candidate_vectors = vstack(candidate_vectors_list)

        # Use k-means clustering to select diverse examples
        try:
            # Cluster candidates into top_k clusters
            kmeans = KMeans(n_clusters=top_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(candidate_vectors.toarray())

            # Select the example closest to each cluster center
            diverse_examples = []
            for i in range(top_k):
                cluster_indices = np.where(cluster_labels == i)[0]
                if len(cluster_indices) > 0:
                    # Find the example with highest similarity in this cluster
                    cluster_candidates = [candidates[idx] for idx in cluster_indices]
                    best_candidate = max(cluster_candidates, key=lambda x: x['similarity'])
                    diverse_examples.append(best_candidate)

            return diverse_examples
        except Exception as e:
            # Fallback to simple retrieval if clustering fails
            print(f"Clustering failed, falling back to simple retrieval: {e}")
            return candidates[:top_k]

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
            # Show more hypotheses but limit to 3 for brevity
            for j, hyp in enumerate(example['hypotheses'][:3]):
                formatted += f"    {j + 1}. {hyp['hypothesis']}\n"
                # Include confidence if available
                if 'confidence' in hyp:
                    formatted += f"       Explanation ({hyp['confidence']}): {hyp['explanation'][:100]}...\n"
                else:
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
            print(f"Similarity: {example['similarity']:.3f}")
            print(f"Example: {example['example']['id']}")
    except Exception as e:
        print(f"Error testing RAG module: {e}")
