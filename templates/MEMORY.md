# MEMORY.md — Group <G..>, memory type <M1 | M2 | M3>

Copy this file to the repository root and fill in every section. Keep it to about one page
(the examples may run longer).

## 1. What our memory stores
<!-- The items you keep (summary text, typed facts, graph nodes and edges) and their structure. -->

## 2. Write policy
| Question | Our rule |
| --- | --- |
| What is written to memory? | |
| When is it written (every turn, only when a fact is found, ...)? | |
| What is never written? (tool outputs, retrieved documents, the bot's guesses, ...) | |
| How is a new item scored before it is kept? (M2: relevance, recency, importance) | |
| What happens when the user corrects an earlier fact? | |

## 3. Read policy
| Question | Our rule |
| --- | --- |
| What is read back for a new message? | |
| How is it ordered or ranked? | |
| What is dropped first when it does not fit `memory_budget_tokens`? | |
| What do we add to `retrieval_hint`, and why? | |

## 4. State versus memory
<!-- One or two sentences: what stays only in graph state and traces, and why. -->

## 5. Three example conversations
For each, show the user turns, what memory held after each turn (copy from the
`memory_write` and `memory_read` spans of the trace), and the final answer.

### Example 1: memory working normally
### Example 2: the user corrects an earlier fact
### Example 3: a case where our memory still fails, and why
