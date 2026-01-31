from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector
from gymnasium import spaces
import random


class GuessGamePettingZooEnv(AECEnv):
    """
    Collaborative guessing game:
    - Each agent guesses an integer in [0, target_range-1].
    - An agent becomes 'correct' once it guesses the target at least once.
    - The episode terminates only when ALL agents are correct.
    - If max_steps is reached, the episode truncates.
    """

    metadata = {"name": "guess_game_pz_v0"}

    def __init__(self, agents, target_range=10, max_steps=10, seed=None):
        super().__init__()

        self.possible_agents = list(agents)
        self.agents = self.possible_agents[:]

        self.target_range = int(target_range)
        self.max_steps = int(max_steps)
        self._rng = random.Random(seed)

        self._agent_selector = AgentSelector(self.possible_agents)

        # Action space: guess integer
        self.action_spaces = {
            agent: spaces.Discrete(self.target_range)
            for agent in self.agents
        }

        # Observation space:
        # - step: current global step
        # - is_correct: whether THIS agent has already guessed correctly at least once
        # - legal_action_mask: optional mask to help avoid repeating known-incorrect guesses
        self.observation_spaces = {
            agent: spaces.Dict({
                "step": spaces.Discrete(self.max_steps + 1),
                "is_correct": spaces.Discrete(2),
                "legal_action_mask": spaces.MultiBinary(self.target_range),
            })
            for agent in self.agents
        }

    def reset(self, seed=None, options=None, target=None, initial_guesses=None):
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        if seed is not None:
            self._rng.seed(seed)
        self.target = target if target is not None else self._rng.randrange(self.target_range)
        self.step_count = 0
        self.last_guesses = {}
        self.correct_flags = {a: False for a in self.agents}
        self.tried_incorrect = set()

        # ✅ 处理初始猜测
        if initial_guesses is not None:
            for agent, guess in initial_guesses.items():
                self.last_guesses[agent] = guess
                if guess == self.target:
                    self.rewards[agent] += 1.0
                    self._cumulative_rewards[agent] += 1.0
                    self.correct_flags[agent] = True
                else:
                    self.tried_incorrect.add(guess)

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        return {a: self.observe(a) for a in self.agents}

    def observe(self, agent):
        # Build legal action mask:
        # - If an agent is already correct, it can still choose any number (mask all 1)
        # - Otherwise, mask out globally known incorrect numbers (0), allow others (1)
        if self.correct_flags[agent]:
            mask = [1] * self.target_range
        else:
            mask = [0 if i in self.tried_incorrect else 1 for i in range(self.target_range)]

        last_guess = self.last_guesses.get(agent, None)
        is_correct_now = self.correct_flags[agent]

        return {
            "step": self.step_count,
            "last_guess": last_guess,
            "is_correct": is_correct_now,
            "legal_action_mask": mask,
        }

    def step(self, action):
        agent = self.agent_selection

        # Dead agent must pass None
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Validate action type for live agent
        if action is None:
            # Live agent cannot pass None; treat as no-op guess (or raise)
            # Here we choose to treat None as no-op: do not change rewards or flags.
            # Alternatively, you could raise ValueError to enforce agent policy.
            pass
        else:
            guess = int(action)
            self.last_guesses[agent] = guess

            # If guessed correctly and not previously correct, award once
            if guess == self.target:
                if not self.correct_flags[agent]:
                    self.rewards[agent] += 1.0
                    self._cumulative_rewards[agent] += 1.0
                self.correct_flags[agent] = True
            else:
                # Record globally known incorrect number
                self.tried_incorrect.add(guess)

        # Termination: only when ALL agents are correct
        if all(self.correct_flags.values()):
            for a in self.agents:
                self.terminations[a] = True

        # Truncation: max_steps reached
        self.step_count += 1
        if self.step_count >= self.max_steps:
            for a in self.agents:
                self.truncations[a] = True

        # Advance to next agent
        self.agent_selection = self._agent_selector.next()

    def render(self):
        print(
            f"[Render] step={self.step_count} target={self.target} "
            f"last_guesses={self.last_guesses} correct_flags={self.correct_flags} "
            f"tried_incorrect={sorted(self.tried_incorrect)}"
        )

    def close(self):
        # No external resources to release in this minimal env
        pass
