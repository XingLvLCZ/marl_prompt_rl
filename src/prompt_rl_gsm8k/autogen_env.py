import asyncio
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from src.prompt_rl_gsm8k.config import AutoGenTask


@dataclass
class AutoGenEpisodeResult:
    task_id: str
    success: bool
    score: float
    rounds: int
    final_answer: str
    transcript: List[str]
    early_stop: bool
    error: Optional[str] = None


class AutoGenEpisodeEnv:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_rounds: int = 6,
        temperature: float = 0.2,
        strategist_name: str = "strategist",
        calculator_name: str = "calculator",
        verifier_name: str = "verifier",
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.strategist_name = strategist_name
        self.calculator_name = calculator_name
        self.verifier_name = verifier_name

    def run_episode(self, protocol_text: str, task: AutoGenTask) -> AutoGenEpisodeResult:
        return asyncio.run(self._run_episode_async(protocol_text=protocol_text, task=task))

    async def _run_episode_async(self, protocol_text: str, task: AutoGenTask) -> AutoGenEpisodeResult:
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import TextMentionTermination
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
                early_stop=False,
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
            model_info={
                "family": "unknown",
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
            },
        )

        strategist_system = (
            "You are the Strategist Agent for GSM8K. Follow the communication protocol strictly. "
            "Your role is to decompose the word problem into ordered subgoals and track unknowns. "
            "Prefer concise, informative communication and adapt detail to uncertainty.\n\n"
            "[COMMUNICATION PROTOCOL]\n"
            f"{protocol_text}"
        )
        calculator_system = (
            "You are the Calculator Agent for GSM8K. Follow the communication protocol strictly. "
            "Your role is to execute arithmetic operations precisely and report intermediate equations. "
            "Prefer concise, informative communication and adapt detail to uncertainty.\n\n"
            "[COMMUNICATION PROTOCOL]\n"
            f"{protocol_text}"
        )
        verifier_system = (
            "You are the Verifier Agent for GSM8K. Follow the communication protocol strictly. "
            "Your role is to cross-check units, consistency, and arithmetic correctness from other agents. "
            "When the team reaches high confidence, provide the final answer using the format **FINAL_ANSWER: <your answer>**.\n\n"
            "[COMMUNICATION PROTOCOL]\n"
            f"{protocol_text}"
        )

        strategist = AssistantAgent(
            name=self.strategist_name,
            model_client=model_client,
            system_message=strategist_system,
        )
        calculator = AssistantAgent(
            name=self.calculator_name,
            model_client=model_client,
            system_message=calculator_system,
        )
        verifier = AssistantAgent(
            name=self.verifier_name,
            model_client=model_client,
            system_message=verifier_system,
        )

        try:
            termination_condition = TextMentionTermination("FINAL_ANSWER")
            team = RoundRobinGroupChat(
                [strategist, calculator, verifier],
                max_turns=self.max_rounds,
                termination_condition=termination_condition,
            )
        except Exception:
            team = RoundRobinGroupChat([strategist, calculator, verifier], max_turns=self.max_rounds)

        try:
            # print(f"\n[Episode Start] Task ID: {task.task_id}, Prompt: {task.prompt}, Expected Answer: {task.expected_answer}\n")
            task_result = await team.run(task=task.prompt)
            transcript = self._extract_transcript(task_result)
            # print(f"\n[Episode End] Task ID: {task.task_id}, Transcript: {transcript}\n")
            final_answer = self._extract_final_answer(transcript)
            effective_rounds = self._effective_rounds(transcript)
            early_stop = effective_rounds < len(transcript)
            success = self._is_correct(final_answer, task.expected_answer)
            score = 1.0 if success else 0.0
            return AutoGenEpisodeResult(
                task_id=task.task_id,
                success=success,
                score=score,
                rounds=effective_rounds,
                final_answer=final_answer,
                transcript=transcript,
                early_stop=early_stop,
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
                early_stop=False,
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
        for line in reversed(transcript):
            match = re.search(r"FINAL_ANSWER\s*:\s*(.+)$", line, flags=re.IGNORECASE)
            if match:
                return AutoGenEpisodeEnv._cleanup_final_answer(match.group(1))

        text = "\n".join(transcript)
        match = re.search(r"FINAL_ANSWER\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
        if match:
            return AutoGenEpisodeEnv._cleanup_final_answer(match.group(1))

        numeric = re.findall(r"-?\d+(?:\.\d+)?", text)
        return numeric[-1] if numeric else ""

    @staticmethod
    def _effective_rounds(transcript: List[str]) -> int:
        if not transcript:
            return 0
        for idx, line in enumerate(transcript, start=1):
            if re.search(r"FINAL_ANSWER\s*:", line, flags=re.IGNORECASE):
                return idx
        return len(transcript)

    @staticmethod
    def _is_correct(pred: str, expected: str) -> bool:
        pred_norm = AutoGenEpisodeEnv._normalize_answer(pred)
        expected_norm = AutoGenEpisodeEnv._normalize_answer(expected)

        if pred_norm.lower() == expected_norm.lower():
            return True

        pred_num = AutoGenEpisodeEnv._to_number(pred_norm)
        expected_num = AutoGenEpisodeEnv._to_number(expected_norm)
        if pred_num is not None and expected_num is not None:
            return abs(pred_num - expected_num) <= 1e-6

        return False

    @staticmethod
    def _normalize_answer(text: str) -> str:
        out = AutoGenEpisodeEnv._cleanup_final_answer(str(text))
        if "####" in out:
            out = out.split("####")[-1].strip()
        out = out.replace(",", "")
        out = out.strip().strip(".")
        return out

    @staticmethod
    def _cleanup_final_answer(text: str) -> str:
        out = str(text).strip()
        out = out.replace("\u200b", "")
        out = re.sub(r"^\s*[-:*]+\s*", "", out)
        out = out.strip().strip("*_` ")
        out = re.sub(r"\*+$", "", out).strip()
        return out

    @staticmethod
    def _to_number(text: str) -> Optional[float]:
        s = text.strip()
        if not s:
            return None
        if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
            try:
                return float(s)
            except ValueError:
                return None

        cleaned = s.replace(",", "")
        numbers = re.findall(r"(?<![\d.])-?\d+(?:\.\d+)?(?![\d.])", cleaned)
        if len(numbers) == 1:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        return None
