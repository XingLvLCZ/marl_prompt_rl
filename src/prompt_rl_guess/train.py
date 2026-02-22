import copy
from collections import deque
import random
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from src.prompt_rl_guess.guess_agent import GuessNumAgent
from src.generator.qwen3_prompt_generator import PromptGenerator
from src.prompt_rl_guess.adaptive_reward import AdaptiveRewardComputer, RewardConfig
from src.prompt_rl_guess.pz_guess_env import GuessGamePettingZooEnv
from src.prompt_rl_guess.pz_guess_runner import GuessGameRunner
from src.generator.protocol_generator import ProtocolGenerator
from src.generator.protocol_loader import ProtocolLoader
from src.provider.qwen import QwenProvider
from src.provider.config import API_KEY, API_URL


NUM_CHOICES = 10
NUM_AGENTS = 2
MAX_STEPS = 20


def _pick_devices():
    """Single-process multi-GPU split.

    - Main trainable prompt generator lives on cuda:0
    - Reference model (for KL) + reward embedding model live on cuda:1 (if available)
    """
    if not torch.cuda.is_available():
        return "cpu", "cpu"
    if torch.cuda.device_count() >= 2:
        return "cuda:0", "cuda:1"
    return "cuda:0", "cuda:0"

def run_episode(protocol: str, agent_provider):
    agents = {
        f"agent_{i}": GuessNumAgent(
            agent_id=f"agent_{i}",
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
    current_target = random.randint(0, NUM_CHOICES - 1)
    print(f"  Target number for this episode: {current_target}")
    trajectory = runner.run_episode(target=current_target)
    return trajectory

train_device, aux_device = _pick_devices()

# Keep the trainable model on a single GPU to make backward/optimizer behavior predictable.
# The second GPU (if present) is used for the frozen reference model and reward embedding model.
prompt_generator = PromptGenerator(
    model_name="/root/aicloud-data/llms/Qwen3-1.7B",
    device_map={"": 0} if train_device.startswith("cuda") else None,
    use_gradient_checkpointing=True,
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
**Generate a protocol-generation prompt** that guides a LLM to generate the protocol.
Reply ONLY the content of the prompt.
"""

prior_knowledge_guidance = """
==================== DESIGN PRINCIPLES (PRIOR KNOWLEDGE) ====================

A high-quality protocol-generation prompt usually includes the following aspects:

1. The prompt should guide the protocol to define information-rich message:
- Historical Information such as guess history: a **list** of previous guesses (int) and a **list** of their correctness (true/false)
- Belief/Inference such as confidence levels and reasoning

2. The prompt should require the protocol to explain "how to process received messages"

3. The prompt focuses on:
- What specific fields enable coordination
- How agents should interpret each other's state
- Example message-response pairs showing state updates

4. The prompt guides LLMs to produce protocols that have:
- Message Schema: JSON structure with 5+ meaningful fields
- Field Semantics: What each field means and how to populate it
- Decision Rules: How to use aggregated information to make next guess
- Example Dialogues: Show message exchanges with state evolution

5. The prompt provides positive and negative examples of messages to illustrate the desired format and common mistakes to avoid.

6. The prompt has length approximately around 700 words, balancing detail and conciseness.

7. The prompt can give LLMs appropriate roles to play when generating the protocol, such as "You are an expert that generates protocols for multi-agent coordination games."
"""

# Prior annealing: progressively summarize prior knowledge over training.
# Stages are defined as fractions of total epochs and mapped to prompt variants.
def build_prior_variants(full_prior: str):
    stripped = "\n".join(line.rstrip() for line in full_prior.strip().splitlines())
    brief = """
==================== DESIGN PRINCIPLES (PRIOR KNOWLEDGE) ====================

A strong **protocol-generation prompt** should:

1. Direct the protocol to define rich messages:
   - Past guesses and correctness  
   - Confidence levels and reasoning  

2. Instruct the protocol on how to process received messages.

3. Emphasize:
   - Fields that enable coordination  
   - How agents interpret each other's state
   - Example message-response pairs  

4. Guide outputs to include:
   - JSON schema with meaningful fields  
   - Clear field semantics  
   - Decision rules for next guesses  
   - Example dialogues showing state changes  

5. Provide positive and negative examples to show correct format and common errors.

6. Be the sufficient and suitable length, balancing detail and clarity.
""".strip()
    compact = """
==================== DESIGN PRINCIPLES (PRIOR KNOWLEDGE) ====================

A well-designed **protocol-generation prompt** should do the following to guide protocol-generation:

- Emphasize creating messages that are informative and interpretable  
- Require clarity on how communication is processed and coordinated  
- Encourage structured outputs with meaningful fields and rules  
- Include illustrative examples to guide correct formatting  
- Balance detail with conciseness
""".strip()
    return {
        "full": stripped,
        "brief": brief,
        "compact": compact,
        "none": "",
    }

def pick_prior_stage(ep: int, total_epochs: int):
    # Fractional schedule: full -> brief -> compact -> none
    progress = (ep + 1) / max(1, total_epochs)
    if progress <= 0.30:
        return "full"
    if progress <= 0.60:
        return "brief"
    if progress <= 0.80:
        return "compact"
    return "none"

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

# LoRA-friendly learning rate schedule: warmup then cosine decay.
BASE_LR = 3e-5
MIN_LR = 1e-5
WARMUP_RATIO = 0.15

optimizer = AdamW(model.parameters(), lr=BASE_LR)

# Initialize adaptive reward computer (keep embedding model off the main training GPU when possible)
reward_device = aux_device if aux_device != train_device else train_device
reward_computer = AdaptiveRewardComputer(config=RewardConfig(device=reward_device))

rewards = []
detailed_scores_history = []
EPOCHS = 100
REF_UPDATE_INTERVAL = 20
reward_window = deque(maxlen=20)

# 在训练循环之前初始化 EMA 参数
ema_mean = 0.0
ema_var = 1.0   # 初始方差不能为0
alpha = 7/(EPOCHS+1)     # 平滑因子，可根据总迭代次数调整（例如 0.05）

# ---------- 初始化自适应 KL ----------
KL_BETA = 0.1
TARGET_KL = 0.01
BETA_MIN = 1e-4
BETA_MAX = 1.0

warmup_epochs = max(1, int(EPOCHS * WARMUP_RATIO))

def lr_lambda(ep: int):
    if ep < warmup_epochs:
        return (ep + 1) / warmup_epochs
    progress = (ep - warmup_epochs) / max(1, (EPOCHS - warmup_epochs))
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)))
    return float((MIN_LR / BASE_LR) + (1.0 - MIN_LR / BASE_LR) * cosine)

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

prior_variants = build_prior_variants(prior_knowledge_guidance)

# Create a slow-moving reference model for KL regularization
ref_model = copy.deepcopy(prompt_generator.model)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad_(False)

# If a second GPU exists, keep the frozen reference model there to reduce peak VRAM on the trainable GPU.
if aux_device != train_device:
    ref_model = ref_model.to(aux_device)

for ep in range(EPOCHS):
    print("=" * 10 + f" Episode {ep+1}/{EPOCHS} " + "=" * 10)
    
    print("generating prompt...")
    prior_stage = pick_prior_stage(ep, EPOCHS)
    prior_text = prior_variants[prior_stage]
    if prior_text:
        full_task_prompt = task_description + "\n\n" + prior_text
    else:
        full_task_prompt = task_description
    prompt, generated_ids, prompt_len, avg_log_prob = prompt_generator.generate_prompt(
        prompt=full_task_prompt,
        temperature=0.7,
        max_new_tokens=7000
    )

    with open(f"src/prompt_rl_guess/prompt_history/prompt_{ep+1}.md", "w") as f:
        f.write(prompt)
    print(
        f"Prompt (len={len(prompt)}) saved at src/prompt_rl_guess/prompt_history/prompt_{ep+1}.md "
        f"[prior_stage={prior_stage}]"
    )

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

    reward_tensor = torch.tensor(reward, dtype=torch.float32, device=avg_log_prob.device)

    with torch.no_grad():
        generated_ids_ref = generated_ids.to(next(ref_model.parameters()).device)
        ref_avg_log_prob = prompt_generator.compute_avg_log_prob(ref_model, generated_ids_ref, prompt_len)
    
    # # Reward whitening using a sliding window of past rewards
    # if len(reward_window) >= 2:
    #     window_mean = sum(reward_window) / len(reward_window)
    #     window_var = sum((r - window_mean) ** 2 for r in reward_window) / (len(reward_window) - 1)
    #     reward_std = (window_var + 1e-8) ** 0.5
    # else:
    #     window_mean = reward
    #     reward_std = 1.0
    # advantage = (reward_tensor - window_mean) / reward_std
    # reward_window.append(reward)

    # 更新 EMA 均值和方差（Welford 在线算法，使用 α 平滑）
    # 均值更新
    delta = reward - ema_mean
    ema_mean = ema_mean + alpha * delta
    # 方差更新（无偏估计的 EMA 版本）
    ema_var = (1 - alpha) * (ema_var + alpha * delta**2)
    reward_std = (ema_var + 1e-8) ** 0.5
    # 计算优势
    advantage = (reward_tensor - ema_mean) / reward_std
    
    kl_div = avg_log_prob - ref_avg_log_prob
    # 自适应 KL 惩罚
    if kl_div.item() > TARGET_KL * 1.5:
        KL_BETA = min(KL_BETA * 1.2, BETA_MAX)
    elif kl_div.item() < TARGET_KL / 1.5:
        KL_BETA = max(KL_BETA / 1.2, BETA_MIN)
    loss = -(avg_log_prob * advantage.detach()) + KL_BETA * kl_div
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    torch.cuda.empty_cache()
    
    print(
        f"[EP {ep + 1}] "
        f"reward={reward:.4f} (game={detailed_scores['game_success']:.4f}, "
        f"protocol={detailed_scores['protocol_quality']:.4f}, "
        f"prompt_q={detailed_scores['prompt_quality']:.4f}), "
        f"steps={len(trajectory)}, "
        f"loss={loss.item():.4f}, "
        f"log_prob={avg_log_prob.item():.4f}, "
        f"kl={kl_div.item():.4f}, "
        f"advantage={advantage.item():.4f}, "
        f"ema_mean={ema_mean:.4f}, "
        f"ema_std={reward_std:.4f}, "
        f"lr={scheduler.get_last_lr()[0]:.6f}\n"
    )
    
    # Print adaptive weights (learned, not fixed)
    print(
        f"  Adaptive Weights: "
        f"w_prompt={detailed_scores['weight_prompt']:.3f}, "
        f"w_protocol={detailed_scores['weight_protocol']:.3f}, "
        f"w_game={detailed_scores['weight_game']:.3f}\n"
    )
    
    # Print semantic evaluation details
    print(
        f"  Semantic Scores: "
        f"prompt[cov={detailed_scores['prompt_coverage']:.3f}, sim={detailed_scores['prompt_similarity']:.3f}], "
        f"protocol[cov={detailed_scores['protocol_coverage']:.3f}, sim={detailed_scores['protocol_similarity']:.3f}]\n"
    )
    
    # Print causal chain correlation estimates
    print(
        f"  Causal Chain: "
        f"prompt\u2192proto={detailed_scores['causal_prompt_proto']:.3f}, "
        f"proto\u2192game={detailed_scores['causal_proto_game']:.3f}, "
        f"game\u2192prompt={detailed_scores['causal_game_prompt']:.3f}\n"
    )

    if (ep + 1) % 10 == 0:
        model.save_pretrained(f"src/prompt_rl_guess/checkpoints/prompt_generator_ep{ep+1}")
        print(f"LoRA adapter saved at episode {ep+1}\n")

    if (ep + 1) % REF_UPDATE_INTERVAL == 0:
        ref_model.load_state_dict(prompt_generator.model.state_dict())
        print(f"Reference model synced at episode {ep + 1}\n")

print(f"\nTraining completed!")
print(f"Average reward over {len(rewards)} episodes: {sum(rewards)/len(rewards):.4f}")
print(f"Max reward: {max(rewards):.4f}")
print(f"Min reward: {min(rewards):.4f}")

if detailed_scores_history:
    avg_prompt_quality = sum(s['prompt_quality'] for s in detailed_scores_history) / len(detailed_scores_history)
    avg_protocol_quality = sum(s['protocol_quality'] for s in detailed_scores_history) / len(detailed_scores_history)
    avg_game_success = sum(s['game_success'] for s in detailed_scores_history) / len(detailed_scores_history)
    
    # Final adaptive weights
    final_weights = detailed_scores_history[-1]
    
    print(f"\n========== AVERAGE METRICS ==========")
    print(f"Prompt Quality:          {avg_prompt_quality:.4f}")
    print(f"Protocol Quality:        {avg_protocol_quality:.4f}")
    print(f"Game Success Rate:       {avg_game_success:.4f}")
    print(f"\n========== FINAL ADAPTIVE WEIGHTS ==========")
    print(f"Weight Prompt:           {final_weights['weight_prompt']:.4f}")
    print(f"Weight Protocol:         {final_weights['weight_protocol']:.4f}")
    print(f"Weight Game:             {final_weights['weight_game']:.4f}")
    print(f"\n========== CAUSAL CHAIN CORRELATIONS ==========")
    print(f"Prompt \u2192 Protocol:     {final_weights['causal_prompt_proto']:.4f}")
    print(f"Protocol \u2192 Game:       {final_weights['causal_proto_game']:.4f}")
    print(f"Game \u2192 Prompt:         {final_weights['causal_game_prompt']:.4f}")
    print(f"====================================\n")
