import re
from pathlib import Path


class ProtocolGenerator:
    """
    Use an LLM to generate a communication protocol in Markdown format.
    """

    def __init__(self, provider):
        self.provider = provider

    def generate_protocol(
        self,
        prompt: str,
        protocol_name: str,
        save_dir: str | None = None,
    ) -> Path:
        """
        Generate a communication protocol using LLM and save it as a Markdown file.
        """
        raw_output = self.provider.call(prompt)

        protocol_markdown = self._extract_protocol_markdown(raw_output)

        spec_dir = Path(__file__).parent / 'specs' if save_dir is None else Path(save_dir)
        Path.mkdir(spec_dir, exist_ok=True)
        
        file_path = spec_dir / f"{protocol_name}.md"
        file_path.write_text(protocol_markdown, encoding="utf-8")

        return file_path

    # ------------------------------------------------------------------
    # Output cleaning & extraction
    # ------------------------------------------------------------------

    def _extract_protocol_markdown(self, llm_output: str) -> str:
        """
        Extract pure Markdown protocol from LLM output.
        Removes <think> blocks and outer ```markdown fences if present.
        """
        cleaned = llm_output.strip()

        # 1️⃣ Remove <think>...</think>
        cleaned = re.sub(
            r".*?</think>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        # 2️⃣ Remove outer ```markdown ... ``` or ``` ... ``` fences
        fence_pattern = r"^```(?:markdown)?\s*(.*?)\s*```$"
        match = re.match(fence_pattern, cleaned, flags=re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()

        # 3️⃣ Final sanity check: must be Markdown
        # if not cleaned.startswith("#"):
        #     raise ValueError(
        #         "Extracted content does not appear to be a valid Markdown protocol.\n"
        #         f"Raw output:\n{llm_output}\n"
        #         f"Cleaned output:\n{cleaned}"
        #     )

        return cleaned


if __name__ == "__main__":
    from src.llm.deepseek import DeepSeekProvider
    from src.llm.config import API_KEY, API_URL

    provider = DeepSeekProvider(api_key=API_KEY, base_url=API_URL, model="deepseek-ai/DeepSeek-V3.1")
    generator = ProtocolGenerator(provider)

    prompt = f"""
You are an expert in multi-agent systems and communication protocol design.

Your task is to generate a COMPLETE communication protocol in **Markdown format**
for a collaborative multi-agent guessing game.

==================== HARD REQUIREMENTS ====================

1. The protocol MUST be written in Markdown.
2. The protocol MUST explicitly require agents to output ONLY valid JSON.
3. Each agent message MUST:
   - Contain a field named: "message_type" with value "state_report"
   - Contain a field named: "next_guess"
4. Agents MUST be instructed NOT to repeat numbers that have already been
   guessed and proven incorrect by any agent.
5. The protocol MUST be suitable for programmatic parsing and reinforcement learning.
6. Do NOT include examples that violate the protocol.
7. Do NOT include explanations outside the protocol.

==================== OUTPUT CONSTRAINT ====================

- Output ONLY the Markdown protocol.
- Do NOT include reasoning, analysis, or <think> blocks.
- Do NOT include any text before or after the protocol.

""".strip()

    protocol_path = generator.generate_protocol(
        prompt=prompt,
        protocol_name="DeepSeek",
    )

    print(f"Protocol saved to: {protocol_path}")