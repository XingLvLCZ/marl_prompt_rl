from src.prompt_rl_autogen.config import AutoGenTask


DEFAULT_PRIOR = """
==================== DESIGN PRINCIPLES (PRIOR KNOWLEDGE) ====================

Design a protocol for two collaborating LLM agents in AutoGen.

1. Define explicit message schema (JSON fields and allowed value types)
2. Define turn-level behavior (what each role should send each round)
3. Define conflict handling (if two candidate answers disagree)
4. Define termination condition (when to emit FINAL_ANSWER)
5. Keep protocol concise, enforce machine-readable communication
""".strip()


HARD_CONSTRAINT = """
==================== HARD REQUIREMENTS ====================

1. Output MUST be a Markdown protocol.
2. Protocol MUST require valid JSON communication between agents.
3. Protocol MUST include a final answer format: FINAL_ANSWER: <value>.
4. Protocol MUST define at least two roles: strategist and solver.

==================== OUTPUT CONSTRAINT ====================

- Output ONLY protocol text.
- Do NOT include analysis or <think> blocks.
""".strip()


def build_protocol_generation_prompt(task: AutoGenTask, prior: str = DEFAULT_PRIOR) -> str:
    return f"""
Task target:
- task_id: {task.task_id}
- collaborative_problem: {task.prompt}

Your task:
Generate a protocol-generation prompt that will guide another LLM to produce a collaboration protocol for AutoGen multi-agent problem solving.
DO NOT solve the problem or give the final answer yourself in the prompt. Focus on designing a high-quality prompt.

{"## Extra information" if prior else ""}
{prior}
""".strip()
