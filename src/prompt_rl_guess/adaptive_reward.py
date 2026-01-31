"""
Adaptive Reward System for Protocol-Generation Prompt Training.

Design Philosophy:
- Use semantic embeddings instead of hard-coded regex patterns
- Let the model learn what constitutes a "good" prompt/protocol through similarity
- Dynamically adjust component weights based on training history
- No fixed magic numbers - everything is learned or derived from data

Key Components:
1. SemanticEvaluator: Uses sentence embeddings to measure quality
2. AdaptiveWeights: Learns optimal weights from game success correlation
3. AdaptiveRewardComputer: Main interface that encapsulates all reward logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class RewardConfig:
    """Configuration for adaptive reward system."""
    # Embedding model for semantic evaluation
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    
    # History buffer for adaptive weight learning
    history_size: int = 50
    
    # Minimum samples before adapting weights
    warmup_episodes: int = 5
    
    # Learning rate for weight adaptation
    weight_lr: float = 0.1
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SemanticEvaluator:
    """
    Evaluate prompt/protocol quality using semantic embeddings.
    
    Instead of regex matching, we compare against "ideal" reference texts
    using cosine similarity. This is more flexible and captures semantic meaning.
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.device = config.device
        self._model = None
        self._tokenizer = None
        
        # Reference embeddings (computed lazily)
        self._prompt_references = None
        self._protocol_references = None
        
        # Reference texts that define "good" prompts/protocols
        self.prompt_reference_texts = [
            # Information richness references
            "The message must include guess_history as a list of all previous guesses",
            "Include feedback_history showing the result of each guess",
            "Add eliminated field listing numbers that have been ruled out",
            "Include remaining_candidates showing possible values",
            "Add confidence score indicating certainty level",
            "Provide reasoning explaining the decision logic",
            
            # Interpretation references
            "Upon receiving a message, update your local state",
            "Add the other agent's eliminated numbers to your exclusion set",
            "Use received information to avoid duplicate guesses",
            "Process incoming messages to update search space",
            
            # Structure references  
            "Define a JSON schema with required fields",
            "Provide example messages showing the format",
            "Include field descriptions and data types",
        ]
        
        self.protocol_reference_texts = [
            # Rich message structure
            "guess_history: array of previous guesses",
            "feedback_history: array of results for each guess",
            "eliminated: numbers ruled out",
            "remaining_candidates: possible values",
            "confidence: float between 0 and 1",
            "reasoning: explanation of decision",
            
            # Interpretation rules
            "Upon receiving a message from another agent",
            "Update your local state based on received information",
            "Add eliminated numbers to your exclusion set",
            "Use confidence to adjust your strategy",
            
            # Decision making
            "Choose next_guess from remaining candidates",
            "Avoid numbers in eliminated list",
            "Use aggregated information to decide",
        ]
    
    def _load_model(self):
        """Lazily load embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.config.embedding_model)
            self._model = self._model.to(self.device)
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for a single text."""
        self._load_model()
        with torch.no_grad():
            embedding = self._model.encode(
                text, 
                convert_to_tensor=True,
                device=self.device
            )
        return embedding
    
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for multiple texts."""
        self._load_model()
        with torch.no_grad():
            embeddings = self._model.encode(
                texts,
                convert_to_tensor=True,
                device=self.device
            )
        return embeddings
    
    def _ensure_references(self):
        """Compute reference embeddings if not already done."""
        if self._prompt_references is None:
            self._prompt_references = self._get_embeddings(self.prompt_reference_texts)
        if self._protocol_references is None:
            self._protocol_references = self._get_embeddings(self.protocol_reference_texts)
    
    def evaluate_prompt(self, prompt: str) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate prompt quality using semantic similarity to references.
        
        Returns:
            Tuple of (overall_score, component_scores)
        """
        self._ensure_references()
        
        # Split prompt into chunks for better matching
        chunks = self._split_into_chunks(prompt, max_chunk_size=200)
        if not chunks:
            return 0.0, {"semantic_coverage": 0.0, "avg_similarity": 0.0}
        
        # Get embeddings for prompt chunks
        chunk_embeddings = self._get_embeddings(chunks)
        
        # Compute similarity matrix: [num_chunks, num_references]
        similarities = F.cosine_similarity(
            chunk_embeddings.unsqueeze(1),  # [C, 1, D]
            self._prompt_references.unsqueeze(0),  # [1, R, D]
            dim=2
        )
        
        # For each reference, take the max similarity across chunks
        # This measures "coverage" - how many reference concepts are present
        max_sims_per_ref = similarities.max(dim=0).values  # [R]
        
        # Coverage: fraction of references that have similarity > threshold
        threshold = 0.5
        coverage = (max_sims_per_ref > threshold).float().mean().item()
        
        # Average similarity: mean of max similarities
        avg_similarity = max_sims_per_ref.mean().item()
        
        # Overall score combines coverage and similarity
        overall = 0.6 * coverage + 0.4 * avg_similarity
        
        return overall, {
            "semantic_coverage": coverage,
            "avg_similarity": avg_similarity,
        }
    
    def evaluate_protocol(self, protocol: str) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate protocol quality using semantic similarity to references.
        """
        self._ensure_references()
        
        chunks = self._split_into_chunks(protocol, max_chunk_size=200)
        if not chunks:
            return 0.0, {"semantic_coverage": 0.0, "avg_similarity": 0.0}
        
        chunk_embeddings = self._get_embeddings(chunks)
        
        similarities = F.cosine_similarity(
            chunk_embeddings.unsqueeze(1),
            self._protocol_references.unsqueeze(0),
            dim=2
        )
        
        max_sims_per_ref = similarities.max(dim=0).values
        
        threshold = 0.5
        coverage = (max_sims_per_ref > threshold).float().mean().item()
        avg_similarity = max_sims_per_ref.mean().item()
        
        overall = 0.6 * coverage + 0.4 * avg_similarity
        
        return overall, {
            "semantic_coverage": coverage,
            "avg_similarity": avg_similarity,
        }
    
    def _split_into_chunks(self, text: str, max_chunk_size: int = 200) -> List[str]:
        """Split text into chunks for embedding."""
        # Split by lines first, then group into chunks
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for line in lines:
            if current_len + len(line) > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [line]
                current_len = len(line)
            else:
                current_chunk.append(line)
                current_len += len(line)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text[:max_chunk_size]]


class AdaptiveWeights(nn.Module):
    """
    Learn optimal weights for combining reward components.
    
    Uses attention mechanism to dynamically weight different components
    based on their correlation with game success.
    """
    
    def __init__(self, num_components: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.num_components = num_components
        
        # Learnable component embeddings
        self.component_embeddings = nn.Parameter(
            torch.randn(num_components, hidden_dim) * 0.1
        )
        
        # Context vector for attention
        self.context = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        
        # History for correlation-based adaptation
        self.score_history = deque(maxlen=50)
        self.success_history = deque(maxlen=50)
        
        # Running correlation estimates
        self.correlations = torch.ones(num_components) / num_components
    
    def forward(self, component_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive weights using attention.
        
        Args:
            component_scores: [batch, num_components] tensor of raw scores
            
        Returns:
            weights: [num_components] tensor of normalized weights
        """
        # Attention scores
        attn_scores = torch.matmul(self.component_embeddings, self.context)
        
        # Incorporate correlation prior
        attn_scores = attn_scores + self.correlations.to(attn_scores.device)
        
        # Softmax to get weights
        weights = F.softmax(attn_scores, dim=0)
        
        return weights
    
    def update_correlations(self, component_scores: Dict[str, float], game_success: float):
        """
        Update correlation estimates based on new episode data.
        """
        scores = list(component_scores.values())[:self.num_components]
        self.score_history.append(scores)
        self.success_history.append(game_success)
        
        if len(self.score_history) >= 10:
            # Compute correlations
            scores_array = np.array(list(self.score_history))
            success_array = np.array(list(self.success_history))
            
            new_corrs = []
            for i in range(min(self.num_components, scores_array.shape[1])):
                if np.std(scores_array[:, i]) > 1e-6 and np.std(success_array) > 1e-6:
                    corr = np.corrcoef(scores_array[:, i], success_array)[0, 1]
                    corr = 0.0 if np.isnan(corr) else corr
                else:
                    corr = 0.0
                new_corrs.append(max(0.0, corr))  # Only positive correlations
            
            # Normalize
            new_corrs = np.array(new_corrs)
            if new_corrs.sum() > 0:
                new_corrs = new_corrs / new_corrs.sum()
            else:
                new_corrs = np.ones(self.num_components) / self.num_components
            
            # Exponential moving average update
            self.correlations = 0.9 * self.correlations + 0.1 * torch.tensor(new_corrs, dtype=torch.float32)


class AdaptiveRewardComputer:
    """
    Main interface for computing adaptive rewards.
    
    Encapsulates all reward computation logic - no external parameters needed.
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.semantic_evaluator = SemanticEvaluator(self.config)
        self.adaptive_weights = AdaptiveWeights(num_components=4)
        
        # Episode counter for warmup
        self.episode_count = 0
        
        # History for baseline computation
        self.reward_history = deque(maxlen=self.config.history_size)
        
        # Component names for logging
        self.component_names = [
            "prompt_quality",
            "protocol_quality", 
            "game_success",
            "length_score"
        ]
    
    def compute_reward(
        self,
        trajectory: List[Dict],
        protocol: str,
        prompt: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute adaptive reward for an episode.
        
        Args:
            trajectory: Game trajectory with agent rewards
            protocol: Generated protocol text
            prompt: Generated prompt text
            
        Returns:
            Tuple of (overall_reward, detailed_scores)
        """
        self.episode_count += 1
        
        # ===== Compute Component Scores =====
        
        # 1. Prompt quality (semantic)
        prompt_score, prompt_details = self.semantic_evaluator.evaluate_prompt(prompt)
        
        # 2. Protocol quality (semantic)
        protocol_score, protocol_details = self.semantic_evaluator.evaluate_protocol(protocol)
        
        # 3. Game success
        game_success = self._compute_game_success(trajectory)
        
        # 4. Length score (normalized, no fixed thresholds)
        length_score = self._compute_adaptive_length_score(prompt)
        
        # ===== Combine with Adaptive Weights =====
        
        component_scores = {
            "prompt_quality": prompt_score,
            "protocol_quality": protocol_score,
            "game_success": game_success,
            "length_score": length_score,
        }
        
        # During warmup, use uniform weights
        if self.episode_count <= self.config.warmup_episodes:
            weights = torch.ones(4) / 4
        else:
            scores_tensor = torch.tensor(list(component_scores.values()))
            weights = self.adaptive_weights(scores_tensor)
        
        # Compute weighted reward
        overall_reward = sum(
            w.item() * s 
            for w, s in zip(weights, component_scores.values())
        )
        
        # Update adaptive weights with game success feedback
        self.adaptive_weights.update_correlations(component_scores, game_success)
        
        # Update history
        self.reward_history.append(overall_reward)
        
        # ===== Build Detailed Scores =====
        
        detailed_scores = {
            "overall": overall_reward,
            # Main components
            "prompt_quality": prompt_score,
            "protocol_quality": protocol_score,
            "game_success": game_success,
            "length_score": length_score,
            # Weights (for monitoring)
            "weight_prompt": weights[0].item(),
            "weight_protocol": weights[1].item(),
            "weight_game": weights[2].item(),
            "weight_length": weights[3].item(),
            # Sub-details
            "prompt_coverage": prompt_details["semantic_coverage"],
            "prompt_similarity": prompt_details["avg_similarity"],
            "protocol_coverage": protocol_details["semantic_coverage"],
            "protocol_similarity": protocol_details["avg_similarity"],
            # Correlation estimates
            "corr_prompt": self.adaptive_weights.correlations[0].item(),
            "corr_protocol": self.adaptive_weights.correlations[1].item(),
            "corr_game": self.adaptive_weights.correlations[2].item(),
            "corr_length": self.adaptive_weights.correlations[3].item(),
        }
        
        return overall_reward, detailed_scores
    
    def _compute_game_success(self, trajectory: List[Dict]) -> float:
        """Compute game success score from trajectory."""
        if not trajectory:
            return 0.0
            
        agent_rewards = {}
        for traj_item in trajectory:
            agent_id = traj_item["agent"]
            if agent_id not in agent_rewards:
                agent_rewards[agent_id] = 0.0
            agent_rewards[agent_id] += traj_item["reward"]
        
        # Fraction of agents that succeeded
        success_count = sum(1 for r in agent_rewards.values() if r > 0)
        return success_count / max(len(agent_rewards), 1)
    
    def _compute_adaptive_length_score(self, prompt: str) -> float:
        """
        Compute length score adaptively based on historical data.
        
        Instead of fixed thresholds, we use the historical distribution
        to normalize the length score.
        """
        current_len = len(prompt)
        
        if not hasattr(self, '_length_history'):
            self._length_history = deque(maxlen=50)
        
        self._length_history.append(current_len)
        
        if len(self._length_history) < 5:
            # During warmup, use a simple heuristic
            # Prefer lengths in 1000-4000 range
            if current_len < 500:
                return current_len / 500 * 0.3
            elif current_len <= 4000:
                return 0.3 + 0.7 * min(1.0, (current_len - 500) / 3500)
            else:
                return max(0.3, 1.0 - (current_len - 4000) / 4000)
        else:
            # Use z-score based on history
            lengths = np.array(list(self._length_history))
            mean_len = lengths.mean()
            std_len = lengths.std() + 1e-6
            
            # Z-score, clipped to [-2, 2]
            z = np.clip((current_len - mean_len) / std_len, -2, 2)
            
            # Convert to [0, 1] score
            # Center (z=0) is best, extremes are worse
            score = 1.0 - abs(z) / 2
            return float(score)
    
    def get_baseline(self) -> float:
        """Get adaptive baseline from reward history."""
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)
    
    def get_summary(self) -> str:
        """Get summary of current adaptive state."""
        weights = self.adaptive_weights.correlations.numpy()
        return (
            f"Adaptive Weights: "
            f"prompt={weights[0]:.3f}, "
            f"protocol={weights[1]:.3f}, "
            f"game={weights[2]:.3f}, "
            f"length={weights[3]:.3f}"
        )


# Convenience function for backward compatibility
def compute_adaptive_reward(
    trajectory: List[Dict],
    protocol: str,
    prompt: str,
    reward_computer: Optional[AdaptiveRewardComputer] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute reward using adaptive system.
    
    Args:
        trajectory: Game trajectory
        protocol: Protocol text
        prompt: Prompt text
        reward_computer: Optional pre-initialized computer (for persistence across episodes)
        
    Returns:
        Tuple of (reward, detailed_scores)
    """
    if reward_computer is None:
        reward_computer = AdaptiveRewardComputer()
    
    return reward_computer.compute_reward(trajectory, protocol, prompt)
