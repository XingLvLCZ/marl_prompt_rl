DEFAULT_PRIOR = """
[DESIGN PRINCIPLES (PRIOR KNOWLEDGE) START]

You are optimizing the quality of a protocol-generation prompt, not directly writing the protocol.
The goal is to produce a prompt that can reliably induce a high-quality collaboration protocol on GSM8K.

Guidance:
- Focus on what makes a prompt induce better collaborative behavior.
- Encourage role specialization, cross-checking, and efficient coordination.
- Prefer flexible but clear communication requirements (avoid overly brittle templates).
- Include evaluation-oriented cues (accuracy, consistency, and turn efficiency).

Prompt quality checklist:
1. Task alignment: clearly anchor to GSM8K-style multi-step arithmetic reasoning.
2. Structural clarity: clear sections, concise instructions, minimal ambiguity.
3. Communication intent: encourage informative and adaptive inter-agent communication.
4. Robustness cues: require self-checking, inconsistency handling, and error recovery.
5. Generalization: avoid brittle, over-specified formatting that harms transfer.
6. Length control: rich enough to guide behavior but not unnecessarily verbose.

[DESIGN PRINCIPLES (PRIOR KNOWLEDGE) END]
""".strip()


DEFAULT_AGENT_STRUCTURE = """
Agent architecture:
- strategist: decomposes the problem, proposes solution plan, tracks unresolved subgoals
- calculator: performs arithmetic operations and returns explicit equation steps
- verifier: validates consistency, checks unit/logic errors, decides when to emit final answer
""".strip()


DEFAULT_TASK_OVERVIEW = """
Dataset overview:
- task_family: GSM8K-style grade-school arithmetic word problems
- objective: maximize final answer correctness under limited interaction rounds
- constraints: communication should be informative, adaptive, and concise
""".strip()


def build_protocol_generation_prompt(
    prior: str = DEFAULT_PRIOR,
    agent_structure: str = DEFAULT_AGENT_STRUCTURE,
    task_overview: str = DEFAULT_TASK_OVERVIEW,
) -> str:
    prompt = f"""
{task_overview}

{agent_structure}

Your task:
Generate a high-quality **protocol-generation prompt** that guides another LLM to produce an effective collaboration protocol.
The generated prompt should emphasize adaptive communication quality over rigid formatting constraints.

{"### Extra Information" if prior else ""}
{prior}
""".strip()
    prompt += "\n\n**OUTPUT ONLY THE PROMPT TEXT**: "
    return prompt