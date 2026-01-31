from src.prompt_rl_guess.pz_guess_env import GuessGamePettingZooEnv
from src.prompt_rl_guess.pz_guess_runner import GuessGameRunner
from src.prompt_rl_guess.guess_agent import GuessNumAgent
from src.llm.qwen import QwenProvider
from src.llm.config import API_KEY, API_URL

# =========================
# 1. Agent & Protocol
# =========================
protocol = """
Message type: state_report

Fields:
- agent_id: string
- next_guess: integer
- reasoning: string

Constraints:
- next_guess must be an integer in the valid range.
- The message must be valid JSON.
"""

provider = QwenProvider(
    api_key=API_KEY,
    base_url=API_URL,
    model="Qwen/Qwen3-14B"
)

NUM_CHOICES = 10

agents = {
    "agent_0": GuessNumAgent(
        agent_id="agent_0",
        provider=provider,
        protocol=protocol,
        initial_guess=0,
        num_choices=NUM_CHOICES,
    ),
    "agent_1": GuessNumAgent(
        agent_id="agent_1",
        provider=provider,
        protocol=protocol,
        initial_guess=1,
        num_choices=NUM_CHOICES,
    ),
}

# =========================
# 2. PettingZoo Env
# =========================

env = GuessGamePettingZooEnv(
    agents=list(agents.keys()),
    target_range=NUM_CHOICES,
    max_steps=10,
)

# =========================
# 3. Runner
# =========================

runner = GuessGameRunner(
    env=env,
    agents=agents
)

# =========================
# 4. Run Episode
# =========================

trajectory = runner.run_episode(target=3)

# =========================
# 5. Print Result
# =========================

print("\n===== EPISODE TRACE =====")
for t, step in enumerate(trajectory):
    print(f"\n--- Step {t + 1} ---")
    print("Agent:", step["agent"])
    print("Observation:", step["observation"])
    print("Messages:", step["messages"])
    print("Action (guess):", step["action"])
    print("Reward:", step["reward"])

print("\n===== DONE =====")
