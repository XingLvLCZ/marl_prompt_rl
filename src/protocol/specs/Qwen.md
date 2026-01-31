# Collaborative Multi-Agent Guessing Game Protocol

## Message Format
All messages must be valid JSON with the following structure:
```json
{
  "message_type": "state_report",
  "next_guess": <number>,
  "guessed_numbers": [
    {"number": <num>, "result": "correct" or "incorrect"},
    ...
  ]
}
```

## Rules
1. **Mandatory Fields**: Each message must contain `message_type` set to `"state_report"` and `next_guess` as a number.
2. **Guessed Numbers**: Agents must include a `guessed_numbers` array listing all previously guessed numbers with their results.
3. **No Repeats**: Agents must not suggest numbers already in `guessed_numbers` with `"result": "incorrect"`.
4. **Correct Guess**: If a number is guessed and proven correct, it must be marked as `"correct"` in the `guessed_numbers` array.
5. **Game Termination**: The game ends when a correct guess is made or all possible numbers are exhausted.