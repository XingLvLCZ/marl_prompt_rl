import torch
from torch.optim import AdamW
from src.prompt_rl_guess.guess_agent import GuessNumAgent
from src.prompt_rl_guess.generator import PromptGenerator
from src.prompt_rl_guess.adaptive_reward import AdaptiveRewardComputer
from src.prompt_rl_guess.pz_guess_env import GuessGamePettingZooEnv
from src.prompt_rl_guess.pz_guess_runner import GuessGameRunner
from src.protocol.generator import ProtocolGenerator
from src.protocol.loader import ProtocolLoader
from src.llm.qwen import QwenProvider
from src.llm.config import API_KEY, API_URL


NUM_CHOICES = 10
NUM_AGENTS = 2
MAX_STEPS = 10
TARGET_NUMBER = 4

def run_episode(protocol: str, agent_provider):
    agents = {
        f"agent_{i}": GuessNumAgent(
            agent_id="agent_0",
            provider=agent_provider,
            protocol=protocol,
            initial_guess=0,
            num_choices=NUM_CHOICES,
        ) for i in range(NUM_AGENTS)
    }
    env = GuessGamePettingZooEnv(
        agents=list(agents.keys()),
        target_range=NUM_CHOICES,
        max_steps=MAX_STEPS,
    )
    runner = GuessGameRunner(
        env=env,
        agents=agents
    )
    trajectory = runner.run_episode(target=TARGET_NUMBER)
    return trajectory

prompt_generator = PromptGenerator(
    model_name="/root/aicloud-data/llms/Qwen3-1.7B",
)
provider = QwenProvider(api_key=API_KEY, base_url=API_URL, model="Qwen/Qwen3-14B")
protocol_generator = ProtocolGenerator(provider=provider)

task_description = f"""
Game:
- game_type: guessing number game
- range_size: {NUM_CHOICES} (from 0 to {NUM_CHOICES - 1})
- num_agents: {NUM_AGENTS}

Game description:
Multiple agents take turns to guess a secret target number within the specified range.
After each guess, agents can follow a protocol and send a message to others.
In the game, agents can ONLY know whether their guess is correct or not, and they CANNOT know if their guess is higher or lower than the target number.

Your task:
Generate a **protocol-generation prompt** that guides a LLM to generate the protocol.
Reply ONLY the content of the **prompt**.
"""

prior_knowledge_guidance = """
==================== DESIGN PRINCIPLES (PRIOR KNOWLEDGE) ====================

A high-quality protocol-generation prompt usually includes the following aspects:

1. The prompt should guide the protocol to define information-rich message:
- Historical Information such as guess history: a **list** of previous guesses and a **list** of their correctness
- Belief/Inference such as confidence levels and reasoning

2. The prompt should require the protocol to explain "how to process received messages"

3. Rather than describing general "coordination strategies", focus on:
- What specific fields enable coordination
- How agents should interpret each other's state
- Example message-response pairs showing state updates

4. A good prompt usually guides LLMs to produce protocols that have:
- Message Schema: JSON structure with 5+ meaningful fields
- Field Semantics: What each field means and how to populate it
- Decision Rules: How to use aggregated information to make next guess
- Example Dialogues: Show message exchanges with state evolution

5. A good prompt usually add explanation in order to control LLMs' behavior.

6. A good prompt usually provides positive and negative examples of messages to illustrate the desired format and common mistakes to avoid.
"""

hard_constraint = """
==================== HARD REQUIREMENTS ====================

1. The protocol MUST be written in Markdown.
2. The protocol MUST explicitly require agents to output ONLY valid JSON.
3. Each agent message MUST contain a field named "next_guess" with an integer value.
4. The message examples MUST be valid JSON without grammar errors although the prompt gives incorrect examples.

==================== OUTPUT CONSTRAINT ====================

- Output ONLY the content of the protocol.
- Do NOT include reasoning, analysis, or <think> blocks.
- Do NOT include any text before or after the protocol.
"""

model = prompt_generator.model
optimizer = AdamW(model.parameters(), lr=1e-4)

# Initialize adaptive reward computer (no external weights needed)
reward_computer = AdaptiveRewardComputer()

baseline = 0.0
rewards = []
detailed_scores_history = []
EPOCHS = 30

for ep in range(EPOCHS):
    print("=" * 10 + f" Episode {ep+1}/{EPOCHS} " + "=" * 10)
    
    print("generating prompt...")
    prompt, log_prob = prompt_generator.generate_prompt(
        prompt=task_description + prior_knowledge_guidance,
        temperature=0.8 - 0.5 * ep / (EPOCHS - 1),  # 从0.8开始，逐渐降到0.3,
        max_new_tokens=10000
    )

    with open(f"src/prompt_rl_guess/prompt_history/prompt_{ep+1}.md", "w") as f:
        f.write(prompt)
    print(f"Prompt (len={len(prompt)}) saved at src/prompt_rl_guess/prompt_history/prompt_{ep+1}.md")

    full_prompt = prompt + "\n\n" + hard_constraint

    print("generating protocol...")
    protocol_name = f"protocol_{ep+1}"
    save_dir = "src/prompt_rl_guess/protocol_history"
    protocol_generator.generate_protocol(
        prompt=full_prompt,
        protocol_name=protocol_name,
        save_dir=save_dir
    )
    protocol = ProtocolLoader.get_protocol_text(
        protocol_name=protocol_name, 
        protocol_dir=save_dir
    )
    print(f"Protocol saved at and loaded from {save_dir}/{protocol_name}.md")

    print("running episode...")
    trajectory = run_episode(protocol, provider)

    # Use adaptive reward computation (no external weights)
    reward, detailed_scores = reward_computer.compute_reward(
        trajectory=trajectory,
        protocol=protocol,
        prompt=prompt
    )
    rewards.append(reward)
    detailed_scores_history.append(detailed_scores)

    reward_tensor = torch.tensor(reward, dtype=torch.float32, device=model.device)
    
    advantage = reward_tensor - baseline
    baseline = 0.9 * baseline + 0.1 * reward
    
    loss = -log_prob * advantage.detach()
    
    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    torch.cuda.empty_cache()
    
    print(
        f"[EP {ep + 1}] "
        f"reward={reward:.4f} (game={detailed_scores['game_success']:.4f}, "
        f"protocol={detailed_scores['protocol_quality']:.4f}, "
        f"prompt_q={detailed_scores['prompt_quality']:.4f}, "
        f"length={detailed_scores['length_score']:.4f}), "
        f"steps={len(trajectory)}, "
        f"loss={loss.item():.4f}, "
        f"log_prob={log_prob.item():.4f}, "
        f"advantage={advantage.item():.4f}, "
        f"baseline={baseline:.4f}\n"
    )
    
    # Print adaptive weights (learned, not fixed)
    print(
        f"  Adaptive Weights: "
        f"w_prompt={detailed_scores['weight_prompt']:.3f}, "
        f"w_protocol={detailed_scores['weight_protocol']:.3f}, "
        f"w_game={detailed_scores['weight_game']:.3f}, "
        f"w_length={detailed_scores['weight_length']:.3f}\n"
    )
    
    # Print semantic evaluation details
    print(
        f"  Semantic Scores: "
        f"prompt[cov={detailed_scores['prompt_coverage']:.3f}, sim={detailed_scores['prompt_similarity']:.3f}], "
        f"protocol[cov={detailed_scores['protocol_coverage']:.3f}, sim={detailed_scores['protocol_similarity']:.3f}]\n"
    )
    
    # Print correlation estimates (how each component relates to game success)
    print(
        f"  Correlations: "
        f"prompt={detailed_scores['corr_prompt']:.3f}, "
        f"protocol={detailed_scores['corr_protocol']:.3f}, "
        f"game={detailed_scores['corr_game']:.3f}, "
        f"length={detailed_scores['corr_length']:.3f}\n"
    )

    if (ep + 1) % 10 == 0:
        model.save_pretrained(f"src/prompt_rl_guess/checkpoints/prompt_generator_ep{ep+1}")
        print(f"LoRA adapter saved at episode {ep+1}\n")

print(f"\nTraining completed!")
print(f"Average reward over {len(rewards)} episodes: {sum(rewards)/len(rewards):.4f}")
print(f"Max reward: {max(rewards):.4f}")
print(f"Min reward: {min(rewards):.4f}")

if detailed_scores_history:
    avg_prompt_quality = sum(s['prompt_quality'] for s in detailed_scores_history) / len(detailed_scores_history)
    avg_protocol_quality = sum(s['protocol_quality'] for s in detailed_scores_history) / len(detailed_scores_history)
    avg_game_success = sum(s['game_success'] for s in detailed_scores_history) / len(detailed_scores_history)
    avg_length_score = sum(s['length_score'] for s in detailed_scores_history) / len(detailed_scores_history)
    
    # Final adaptive weights
    final_weights = detailed_scores_history[-1]
    
    print(f"\n========== AVERAGE METRICS ==========")
    print(f"Prompt Quality:          {avg_prompt_quality:.4f}")
    print(f"Protocol Quality:        {avg_protocol_quality:.4f}")
    print(f"Game Success Rate:       {avg_game_success:.4f}")
    print(f"Length Score:            {avg_length_score:.4f}")
    print(f"\n========== FINAL ADAPTIVE WEIGHTS ==========")
    print(f"Weight Prompt:           {final_weights['weight_prompt']:.4f}")
    print(f"Weight Protocol:         {final_weights['weight_protocol']:.4f}")
    print(f"Weight Game:             {final_weights['weight_game']:.4f}")
    print(f"Weight Length:           {final_weights['weight_length']:.4f}")
    print(f"\n========== LEARNED CORRELATIONS ==========")
    print(f"Corr Prompt→Success:     {final_weights['corr_prompt']:.4f}")
    print(f"Corr Protocol→Success:   {final_weights['corr_protocol']:.4f}")
    print(f"Corr Game→Success:       {final_weights['corr_game']:.4f}")
    print(f"Corr Length→Success:     {final_weights['corr_length']:.4f}")
    print(f"====================================\n")
