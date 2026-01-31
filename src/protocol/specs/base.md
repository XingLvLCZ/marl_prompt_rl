# Guess Number Game Communication Protocol

## Protocol Metadata
- **Name**: GuessNumberGameProtocol
- **Version**: 1.0
- **Purpose**: Multi-agent collaborative guessing game with feedback-driven learning
- **Description**: Agents communicate their state and proposed guesses using a structured message format

## Message Types

### Message Type: state_report
- **Description**: Agent reports its current state in the game and proposes next guess
- **Triggers**: after_game_step, at_round_start
- **Answer Field**: next_guess

#### Required Fields
| Field Name | Type | Description |
|-----------|------|-------------|
| sender | string | Agent identifier |
| type | string | Message type (must be "state_report") |
| timestamp | integer | Game step number |
| next_guess | integer | Proposed next guess (answer_field) |

#### Optional Fields
| Field Name | Type | Description |
|-----------|------|-------------|
| content | object | Message content containing game state details |
| reasoning | string | LLM-generated reasoning |
| confidence | float | Confidence in guess (0-1) |

#### Constraints
- sender must be a valid agent_id
- type must equal "state_report"
- timestamp >= 0
- 0 <= next_guess < num_choices

#### Example
```json
{
  "sender": "agent_0",
  "type": "state_report",
  "timestamp": 2,
  "next_guess": 3,
  "content": {
    "current_guess": 3,
    "is_correct": false,
    "guess_history": [0, 2, 3],
    "feedback_history": [false, false]
  },
  "reasoning": "Based on feedback from previous guesses, trying next option.",
  "confidence": 0.6
}
```

### Message Type: proposal
- **Description**: Agent proposes the next guess
- **Triggers**: communication_phase

#### Required Fields
| Field Name | Type | Description |
|-----------|------|-------------|
| content | object | Proposal content |

#### Optional Fields
| Field Name | Type | Description |
|-----------|------|-------------|
| rationale | string | Reasoning for the proposal |

---

## Communication Phases

### Phase 1: Observation Phase
- **Description**: Agents receive environment feedback
- **Duration**: 1 round
- **Involved Actors**: env, agents
- **Mandatory Message**: None

### Phase 2: Reporting Phase
- **Description**: All agents send state reports
- **Duration**: 1 round
- **Involved Actors**: agents
- **Mandatory Message**: state_report

### Phase 3: Reasoning Phase
- **Description**: Agents reason about next guess based on reports
- **Duration**: 1 round
- **Involved Actors**: agents, llm
- **Mandatory Message**: None

### Phase 4: Action Phase
- **Description**: Agents execute next action (make a guess)
- **Duration**: 1 round
- **Involved Actors**: agents
- **Mandatory Message**: None

---

## Global Validation Rules
1. All messages must include sender, type, timestamp
2. sender must be a known agent_id
3. type must be one of: state_report, proposal
4. timestamp must be a non-negative integer
5. content must include all required fields for the message type
6. Values in content must satisfy message type constraints
