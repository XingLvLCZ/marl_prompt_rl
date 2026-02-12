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
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    # embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 1024
    
    # History buffer for adaptive weight learning
    history_size: int = 100
    
    # Minimum samples before adapting weights
    warmup_episodes: int = 10
    
    # Learning rate for weight adaptation
    weight_lr: float = 0.05
    
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
        # Prompt references: actual instructional content that would appear in generated prompts
        self.prompt_reference_texts = [
            # Instructing protocol to include information-rich messages
            "Design a JSON message schema with the following required fields for agent communication",
            "Each message must include a guess_history field containing the list of all previous guesses",
            "Include a feedback_history field showing whether each guess was correct or incorrect",
            "Add a confidence field indicating the agent's certainty level about its next guess",
            "Include a reasoning field explaining the logic behind the decision",
            
            # Instructing how to process messages
            "Explain how agents should process received messages and update their local state",
            "Describe the steps for integrating information from other agents",
            "Specify how agents should use received guess history to avoid redundant attempts",
            
            # Instructing coordination mechanisms
            "Define which message fields enable effective coordination between agents",
            "Explain how agents should interpret and utilize each other's confidence scores",
            "Provide clear rules for making decisions based on aggregated multi-agent information",
            
            # Instructing structure and examples
            "Include at least 5 meaningful fields in the message schema",
            "Provide example message-response pairs demonstrating how state evolves during coordination",
            "Show positive examples of well-structured messages that enable successful cooperation",
            "Show negative examples of poorly-designed messages that hinder coordination",
            "Describe common mistakes in protocol design and how to avoid them",
        ]
        
        # === Negative references: what BAD prompts look like ===
        self.prompt_negative_texts = [
            "Generate a protocol for agents to communicate",
            "Create a simple message format for the game",
            "Agents should coordinate and share information",
            "Design a communication protocol",
            "Make a JSON format for guessing",
            "The protocol should be clear and effective",
        ]
        
        # === Negative references: what BAD protocols look like ===
        self.protocol_negative_texts = [
            "Send your guess to the other agent",
            "Message contains the next guess",
            "Status correct or incorrect",
            "Attempted list of guessed numbers",
            "Agents take turns guessing",
            "The game ends when someone guesses correctly",
        ]
        
        # Negative embeddings (computed lazily)
        self._prompt_negatives = None
        self._protocol_negatives = None
        
        # Dynamic thresholds (computed from reference statistics)
        self._prompt_pos_threshold = None
        self._prompt_neg_threshold = None
        self._protocol_pos_threshold = None
        self._protocol_neg_threshold = None
        
        # Protocol references: actual content that would appear in generated protocols
        self.protocol_reference_texts = [
            # Message schema definition (what appears in protocol)
            "Message Schema: guess_history array, feedback_history array, confidence float, reasoning string, next_guess integer",
            "guess_history: An array of integers representing all previous guesses made by this agent",
            "feedback_history: An array of booleans indicating whether each guess was correct",
            "confidence: A float value between 0 and 1 representing certainty about the next guess",
            "reasoning: A string explaining the logic and inference behind the next guess",
            "next_guess: An integer representing the agent's next guess within the valid range",
            
            # Message processing instructions (what appears in protocol)
            "Upon receiving a message from another agent extract their guess_history and feedback_history",
            "Add the sender's incorrect guesses to your local exclusion set",
            "Update your belief about remaining candidates by removing eliminated numbers",
            "Weight the sender's information by their confidence score when making decisions",
            "Maintain a combined view of all agents' exploration history",
            
            # Decision rules (what appears in protocol)
            "Calculate remaining_candidates by excluding all numbers with false feedback",
            "Select next_guess from remaining_candidates to maximize information gain",
            "If multiple agents have high confidence in different numbers avoid guessing the same",
            "Coordinate to ensure diverse exploration of the search space",
            
            # Example exchanges (what appears in protocol)
            "Example: Agent A sends guess_history=[2,5] feedback_history=[false,false] confidence=0.7 next_guess=3",
            "Example: Agent B receives this message and excludes 2 and 5 from candidates",
            "Example showing incorrect message: missing next_guess field causes coordination failure",
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
        """Compute reference embeddings and dynamic thresholds if not already done."""
        if self._prompt_references is None:
            self._prompt_references = self._get_embeddings(self.prompt_reference_texts)
        if self._protocol_references is None:
            self._protocol_references = self._get_embeddings(self.protocol_reference_texts)
        if self._prompt_negatives is None:
            self._prompt_negatives = self._get_embeddings(self.prompt_negative_texts)
        if self._protocol_negatives is None:
            self._protocol_negatives = self._get_embeddings(self.protocol_negative_texts)
        
        # Compute dynamic thresholds once
        if self._prompt_pos_threshold is None:
            self._compute_dynamic_thresholds()
    
    def _compute_dynamic_thresholds(self):
        """
        Compute thresholds dynamically from reference embedding statistics.
        
        Positive threshold: derived from within-group pairwise similarity
          of positive references. A chunk "covers" a reference if its similarity
          exceeds the average similarity between reference concepts.
          
        Negative threshold: derived from cross-group similarity between
          positive and negative references. Penalize only when similarity
          to a negative reference is clearly above the positive-negative noise floor.
        """
        # --- Prompt thresholds ---
        prompt_pos_pw = self._pairwise_similarities(self._prompt_references)
        prompt_cross = self._cross_similarities(self._prompt_references, self._prompt_negatives)
        
        # Positive: mean pairwise similarity among positives
        # A chunk must be at least as similar to a reference as references are to each other
        prompt_pos_mean = prompt_pos_pw.mean().item()
        prompt_pos_std = prompt_pos_pw.std().item()
        self._prompt_pos_threshold = prompt_pos_mean
        
        # Negative: cross-group mean + 1.5*std (penalize only clear matches to bad patterns)
        cross_mean = prompt_cross.mean().item()
        cross_std = prompt_cross.std().item()
        self._prompt_neg_threshold = cross_mean + 1.5 * cross_std
        
        # --- Protocol thresholds ---
        proto_pos_pw = self._pairwise_similarities(self._protocol_references)
        proto_cross = self._cross_similarities(self._protocol_references, self._protocol_negatives)
        
        proto_pos_mean = proto_pos_pw.mean().item()
        proto_pos_std = proto_pos_pw.std().item()
        self._protocol_pos_threshold = proto_pos_mean
        
        proto_cross_mean = proto_cross.mean().item()
        proto_cross_std = proto_cross.std().item()
        self._protocol_neg_threshold = proto_cross_mean + 1.5 * proto_cross_std
        
        print(f"[Dynamic Thresholds] Prompt:  pos={self._prompt_pos_threshold:.4f}  neg={self._prompt_neg_threshold:.4f}")
        print(f"[Dynamic Thresholds] Protocol: pos={self._protocol_pos_threshold:.4f}  neg={self._protocol_neg_threshold:.4f}")
        print(f"  (prompt refs pairwise: mean={prompt_pos_mean:.4f} std={prompt_pos_std:.4f})")
        print(f"  (protocol refs pairwise: mean={proto_pos_mean:.4f} std={proto_pos_std:.4f})")
    
    def _pairwise_similarities(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute upper-triangle pairwise cosine similarities (excluding self-pairs)."""
        n = embeddings.shape[0]
        sim_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
        # Extract upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=sim_matrix.device), diagonal=1)
        return sim_matrix[mask]
    
    def _cross_similarities(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """Compute all cross-group cosine similarities between two sets."""
        # [len_a, len_b]
        sim_matrix = F.cosine_similarity(
            emb_a.unsqueeze(1), emb_b.unsqueeze(0), dim=2
        )
        return sim_matrix.flatten()
    
    def evaluate_prompt(self, prompt: str) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate prompt quality using semantic similarity to references.
        
        Scoring:
        1. Positive coverage: fraction of references well-matched (dynamic threshold)
        2. Negative penalty: similarity to known-bad patterns (dynamic threshold)
        3. Specificity: variance of per-reference similarities
           High = targeted content (good), Low = vague/generic (bad)
        
        Returns:
            Tuple of (overall_score, component_scores)
        """
        self._ensure_references()
        
        # Split prompt into chunks for better matching
        chunks = self._split_into_chunks(prompt, max_chunk_size=200)
        if not chunks:
            return 0.0, {"semantic_coverage": 0.0, "avg_similarity": 0.0,
                         "negative_penalty": 0.0, "specificity": 0.0}
        
        chunk_embeddings = self._get_embeddings(chunks)
        
        # === Positive scoring (dynamic threshold) ===
        similarities = F.cosine_similarity(
            chunk_embeddings.unsqueeze(1),
            self._prompt_references.unsqueeze(0),
            dim=2
        )
        max_sims_per_ref = similarities.max(dim=0).values
        
        pos_threshold = self._prompt_pos_threshold
        coverage = (max_sims_per_ref > pos_threshold).float().mean().item()
        avg_similarity = max_sims_per_ref.mean().item()
        
        # === Negative penalty (dynamic threshold) ===
        neg_similarities = F.cosine_similarity(
            chunk_embeddings.unsqueeze(1),
            self._prompt_negatives.unsqueeze(0),
            dim=2
        )
        # If any chunk is very similar to a negative reference, penalize
        max_neg_per_ref = neg_similarities.max(dim=0).values
        neg_penalty = (max_neg_per_ref > self._prompt_neg_threshold).float().mean().item()
        
        # === Specificity: std of per-reference max similarities ===
        # High std means the prompt specifically addresses some concepts
        # Low std means the prompt is vaguely similar to everything (generic)
        specificity_raw = max_sims_per_ref.std().item()
        # Normalize: typical std range is [0.02, 0.12], map to [0, 1]
        specificity = min(1.0, specificity_raw / 0.10)
        
        # === Combine ===
        positive_score = 0.4 * coverage + 0.3 * avg_similarity + 0.3 * specificity
        overall = max(0.0, positive_score - 0.3 * neg_penalty)
        
        return overall, {
            "semantic_coverage": coverage,
            "avg_similarity": avg_similarity,
            "negative_penalty": neg_penalty,
            "specificity": specificity,
        }
    
    def evaluate_protocol(self, protocol: str) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate protocol quality using semantic similarity to references.
        
        Same structure as prompt evaluation:
        positive coverage + negative penalty + specificity.
        """
        self._ensure_references()
        
        chunks = self._split_into_chunks(protocol, max_chunk_size=200)
        if not chunks:
            return 0.0, {"semantic_coverage": 0.0, "avg_similarity": 0.0,
                         "negative_penalty": 0.0, "specificity": 0.0}
        
        chunk_embeddings = self._get_embeddings(chunks)
        
        # === Positive scoring (dynamic threshold) ===
        similarities = F.cosine_similarity(
            chunk_embeddings.unsqueeze(1),
            self._protocol_references.unsqueeze(0),
            dim=2
        )
        max_sims_per_ref = similarities.max(dim=0).values
        
        pos_threshold = self._protocol_pos_threshold
        coverage = (max_sims_per_ref > pos_threshold).float().mean().item()
        avg_similarity = max_sims_per_ref.mean().item()
        
        # === Negative penalty (dynamic threshold) ===
        neg_similarities = F.cosine_similarity(
            chunk_embeddings.unsqueeze(1),
            self._protocol_negatives.unsqueeze(0),
            dim=2
        )
        max_neg_per_ref = neg_similarities.max(dim=0).values
        neg_penalty = (max_neg_per_ref > self._protocol_neg_threshold).float().mean().item()
        
        # === Specificity: std of per-reference max similarities ===
        specificity_raw = max_sims_per_ref.std().item()
        specificity = min(1.0, specificity_raw / 0.10)
        
        # === Combine ===
        positive_score = 0.4 * coverage + 0.3 * avg_similarity + 0.3 * specificity
        overall = max(0.0, positive_score - 0.3 * neg_penalty)
        
        return overall, {
            "semantic_coverage": coverage,
            "avg_similarity": avg_similarity,
            "negative_penalty": neg_penalty,
            "specificity": specificity,
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
    
    def __init__(self, num_components: int = 3, hidden_dim: int = 32):
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
        self.adaptive_weights = AdaptiveWeights(num_components=3)
        
        # Episode counter for warmup
        self.episode_count = 0
        
        # History for baseline computation
        self.reward_history = deque(maxlen=self.config.history_size)
        
        # Component names for logging
        self.component_names = [
            "prompt_quality",
            "protocol_quality", 
            "game_success",
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
        
        # ===== Combine with Adaptive Weights =====
        
        component_scores = {
            "prompt_quality": prompt_score,
            "protocol_quality": protocol_score,
            "game_success": game_success,
        }
        
        # During warmup, use uniform weights
        if self.episode_count <= self.config.warmup_episodes:
            weights = torch.ones(3) / 3
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
            # Weights (for monitoring)
            "weight_prompt": weights[0].item(),
            "weight_protocol": weights[1].item(),
            "weight_game": weights[2].item(),
            # Sub-details
            "prompt_coverage": prompt_details["semantic_coverage"],
            "prompt_similarity": prompt_details["avg_similarity"],
            "prompt_neg_penalty": prompt_details["negative_penalty"],
            "prompt_specificity": prompt_details["specificity"],
            "protocol_coverage": protocol_details["semantic_coverage"],
            "protocol_similarity": protocol_details["avg_similarity"],
            "protocol_neg_penalty": protocol_details["negative_penalty"],
            "protocol_specificity": protocol_details["specificity"],
            # Correlation estimates
            "corr_prompt": self.adaptive_weights.correlations[0].item(),
            "corr_protocol": self.adaptive_weights.correlations[1].item(),
            "corr_game": self.adaptive_weights.correlations[2].item(),
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
            f"game={weights[2]:.3f}"
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


if __name__ == "__main__":
    # 初始化奖励计算器
    reward_computer = AdaptiveRewardComputer()

    # 构造一个简单的测试数据
    trajectory = [
        {"agent": "A", "reward": 1.0},
        {"agent": "B", "reward": 0.0},
        {"agent": "C", "reward": 1.0},
    ]
    protocol = "Message Schema: guess_history, feedback_history, confidence, reasoning, next_guess"
    prompt = "Design a JSON message schema with required fields for agent communication"

    # 计算奖励
    reward, details = reward_computer.compute_reward(trajectory, protocol, prompt)

    # 打印结果
    print("=== Test Run ===")
    print(f"Reward: {reward:.4f}")
    print("Details:")
    for k, v in details.items():
        print(f"  {k}: {v}")
