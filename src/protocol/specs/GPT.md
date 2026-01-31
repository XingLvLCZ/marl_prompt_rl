# Collaborative Guessing Game Communication Protocol

## 1. Overview

This protocol defines how agents communicate in a collaborative multi-agent guessing game.

All agents cooperate to identify a hidden target number within a known range.
The game ends only when **all agents have guessed the correct number**.

Agents MUST communicate their guesses, outcomes, and reasoning-relevant state
to ensure efficient coverage of the search space and to avoid redundant or
repeated incorrect guesses.

---

## 2. General Rules (MANDATORY)

1. Agents MUST output **ONLY valid JSON**.
2. Any output that is not valid JSON is considered invalid.
3. Agents MUST follow the message schemas defined in this protocol.
4. Agents MUST include a `next_guess` field indicating their chosen number.
5. Agents MUST consider other agents’ previous guesses and outcomes.
6. Agents MUST NOT repeat numbers that have already been proven incorrect
   by themselves or by other agents.

---

## 3. Message Types

### 3.1 `state_report`

Used by an agent to broadcast its current guess and local state.

#### Schema

```json
{
  "type": "state_report",
  "agent_id": "string",
  "step": "integer",
  "next_guess": "integer",
  "is_correct": "boolean",
  "guess_history": "integer[]",
  "feedback_history": "boolean[]"
}
