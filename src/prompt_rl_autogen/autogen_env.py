import asyncio
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from src.prompt_rl_autogen.config import AutoGenTask


@dataclass
class AutoGenEpisodeResult:
    task_id: str
    success: bool
    score: float
    rounds: int
    final_answer: str
    transcript: List[str]
    error: Optional[str] = None


class AutoGenEpisodeEnv:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_rounds: int = 6,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_rounds = max_rounds
        self.temperature = temperature

    def run_episode(self, protocol_text: str, task: AutoGenTask) -> AutoGenEpisodeResult:
        return asyncio.run(self._run_episode_async(protocol_text=protocol_text, task=task))

    async def _run_episode_async(self, protocol_text: str, task: AutoGenTask) -> AutoGenEpisodeResult:
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.teams import RoundRobinGroupChat
            from autogen_ext.models.openai import OpenAIChatCompletionClient
        except Exception as exc:
            return AutoGenEpisodeResult(
                task_id=task.task_id,
                success=False,
                score=-1.0,
                rounds=0,
                final_answer="",
                transcript=[],
                error=(
                    "AutoGen dependencies not available. Install: "
                    "pip install autogen-agentchat autogen-ext[openai]"
                    f" | detail: {exc}"
                ),
            )

        model_client = OpenAIChatCompletionClient(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
        )

        strategist_system = (
            "You are the Strategy Agent. Follow the communication protocol strictly. "
            "You must produce concise coordination messages and avoid solving alone.\n\n"
            "[COMMUNICATION PROTOCOL]\n"
            f"{protocol_text}"
        )
        solver_system = (
            "You are the Solver Agent. Follow the communication protocol strictly. "
            "Use received strategy messages to produce the final answer. "
            "Final output must include: FINAL_ANSWER: <value>.\n\n"
            "[COMMUNICATION PROTOCOL]\n"
            f"{protocol_text}"
        )

        strategist = AssistantAgent(
            name="strategist",
            model_client=model_client,
            system_message=strategist_system,
        )
        solver = AssistantAgent(
            name="solver",
            model_client=model_client,
            system_message=solver_system,
        )

        team = RoundRobinGroupChat([strategist, solver], max_turns=self.max_rounds)

        try:
            task_result = await team.run(task=task.prompt)
            transcript = self._extract_transcript(task_result)
            final_answer = self._extract_final_answer(transcript)
            success = self._is_correct(final_answer, task.expected_answer)
            score = 1.0 if success else 0.0
            rounds = len(transcript)
            return AutoGenEpisodeResult(
                task_id=task.task_id,
                success=success,
                score=score,
                rounds=rounds,
                final_answer=final_answer,
                transcript=transcript,
                error=None,
            )
        except Exception as exc:
            return AutoGenEpisodeResult(
                task_id=task.task_id,
                success=False,
                score=-1.0,
                rounds=0,
                final_answer="",
                transcript=[],
                error=str(exc),
            )
        finally:
            if hasattr(model_client, "close"):
                await model_client.close()

    @staticmethod
    def _extract_transcript(task_result: Any) -> List[str]:
        messages = getattr(task_result, "messages", [])
        transcript: List[str] = []
        for item in messages:
            source = getattr(item, "source", "agent")
            content = getattr(item, "content", "")
            if isinstance(content, list):
                content = "\n".join(str(x) for x in content)
            transcript.append(f"[{source}] {str(content).strip()}")
        return transcript

    @staticmethod
    def _extract_final_answer(transcript: List[str]) -> str:
        text = "\n".join(transcript)
        match = re.search(r"FINAL_ANSWER\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        numeric = re.findall(r"-?\d+(?:\.\d+)?", text)
        return numeric[-1] if numeric else ""

    @staticmethod
    def _is_correct(pred: str, expected: str) -> bool:
        return pred.strip().lower() == expected.strip().lower()
