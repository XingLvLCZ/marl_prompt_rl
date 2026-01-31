class GuessGameRunner:
    """
    Runner that:
    - Resets env and agents
    - For each agent turn:
        * observe
        * run communication rounds
        * act (or None if dead)
        * step env
    - Stops when all terminations or all truncations
    """

    def __init__(self, env, agents):
        self.env = env
        self.agents = agents  # dict: {agent_id: GuessNumAgent or compatible}

    def run_episode(self, target=None):
        # Reset env with optional target
        initial_guesses = {aid: ag.initial_guess for aid, ag in self.agents.items()}
        observations = self.env.reset(target=target, initial_guesses=initial_guesses)
        # print(observations)

        # Sync per-agent config
        num_agents = len(self.agents)
        # If your GuessNumAgent needs num_choices, you can pass env.target_range here
        num_choices = getattr(self.env, "target_range", None)

        for agent_id, agent in self.agents.items():
            agent.reset()
            agent.num_agents = num_agents
            if num_choices is not None and hasattr(agent, "num_choices"):
                agent.num_choices = num_choices

        trajectory = []

        # Iterate PettingZoo agent turns
        for agent_id in self.env.agent_iter():
            agent = self.agents[agent_id]

            # Pull latest observation for this agent
            obs = self.env.observe(agent_id)
            agent.update_observation(obs)

            # ===== Communication phase =====
            msg = agent.send_message()

            # Broadcast messages to other agents
            for other_id, other in self.agents.items():
                if other_id != agent_id:
                    other.receive_messages(msg)

            # ===== Action =====
            # If agent is dead (terminated or truncated), must pass None
            if self.env.terminations.get(agent_id, False) or self.env.truncations.get(agent_id, False):
                action = None
            else:
                action = agent.act()

            # Step environment
            self.env.step(action)

            # Record trajectory safely
            traj_item = {
                "agent": agent_id,
                "observation": obs,
                "action": action,
                "messages": msg,
                "reward": self.env.rewards.get(agent_id, 0.0),
                "done": self.env.terminations.get(agent_id, False) or self.env.truncations.get(agent_id, False),
            }
            trajectory.append(traj_item)

            # Stop if episode ended (all terminations or all truncations)
            if all(self.env.terminations.values()) or all(self.env.truncations.values()):
                break

        return trajectory
