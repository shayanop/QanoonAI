# QanoonArena file formats

This file defines every format you read or write in Assignment 3. The checker scripts
enforce these rules, so if a script says your file is OK, the format is right.

Real, ready-made examples of every question type are in
`harness/data/example_sessions.jsonl`. Open that file next to this one.

---

## 1. Document ids

Every document in `harness/corpus/` has an id. Your chatbot cites these ids, and
your test questions use them.

| Kind | Id pattern | Example |
| --- | --- | --- |
| Statute section | `ACT-SECTION` | `PPC-379`, `PPC-489F`, `CPC-151`, `CA-73` |
| Sindh High Court judgment | `SHC-` + case code | `SHC-140894` |

The corpus has seven Acts: Pakistan Penal Code, 1860 (`PPC`), Code of Civil Procedure, 1908 (`CPC`),
Contract Act, 1872 (`CA`), Limitation Act, 1908 (`LA`), Specific Relief Act, 1877 (`SRA`), Transfer of
Property Act, 1882 (`TPA`) and Arms Act, 1878 (`AA`), from the official Pakistan Code texts; and a set of
Sindh High Court judgments. Amended sections end with their official amendment notes, for example
`[Amendment notes: Ins. by Ord. 85 of 2002, s.2.]`.

To see all ids: `python -c "from harness.common import Corpus; print(sorted(Corpus().docs)[:50])"`

---

## 2. Question sessions (`.jsonl`, one session per line)

A **session** is one conversation of 1 to 6 user turns. Only the turns that have a
`gold` record are marked; the other turns just give the bot information.

```json
{
  "session_id": "G07-T3-01",
  "type": "T3",
  "author": "G07",
  "turns": [
    {"user": "Someone took my mobile phone from my shop on 3 March 2025.", "gold": null},
    {"user": "I know who did it. It was my neighbour.", "gold": null},
    {"user": "We tried to settle it ourselves but it did not work.", "gold": null},
    {"user": "Based on what I told you, which section of the Pakistan Penal Code, 1860 sets the punishment for this?",
     "gold": {
       "answer": "Section 379 of the Pakistan Penal Code, 1860 sets the punishment for theft.",
       "answer_keywords": ["379"],
       "citations": ["PPC-379"],
       "exact_version": false,
       "abstain": false,
       "memory_dependent": true,
       "depends_on_turns": [1],
       "as_of": null,
       "evidence": [{"doc_id": "PPC-379",
                     "quote": "Whoever commits theft shall be punished with imprisonment of either description"}]
     }}
  ]
}
```

(The quote above shows the shape only. Your quotes must be copied word for word from
the corpus file, and the checker verifies this.)

### Session fields

| Field | Required | Meaning |
| --- | --- | --- |
| `session_id` | yes | Unique. For Step 1 use `<group>-<type>-<nn>`, e.g. `G07-T3-01` |
| `type` | yes | One of `T1` to `T5` (see the table below) |
| `author` | yes | Your group id |
| `turns` | yes | 1 to 6 turns; T3 and T4 need at least 2 |
| `turns[].user` | yes | What the user types |
| `turns[].gold` | for marked turns | The correct answer record, or `null` |

### Gold fields

| Field | Type | Meaning |
| --- | --- | --- |
| `answer` | text | A short reference answer in plain English |
| `answer_keywords` | list of text | 1 to 4 words or numbers that **every** correct answer must contain, e.g. the section number. Write alternatives with a bar: `"dismissed\|rejected"`. May be empty only when `abstain` is true |
| `citations` | list of ids | The documents a correct answer must cite. Empty only when `abstain` is true |
| `exact_version` | true/false | Leave `false` (reserved) |
| `abstain` | true/false | `true` when the right reply is "I am not sure" (all T5 turns) |
| `memory_dependent` | true/false | `true` when the turn cannot be answered without an earlier turn |
| `depends_on_turns` | list of numbers | Which earlier turns (1 = first) it needs; required when `memory_dependent` is true |
| `as_of` | `YYYY-MM-DD` or null | Optional: a date the question is about; usually `null` |
| `evidence` | list | `{doc_id, quote}` pairs: the exact words in the corpus that prove the answer, at least 20 characters. Not needed when `abstain` is true |

### The five types

| Type | What it tests | Rules the checker enforces |
| --- | --- | --- |
| T1 Statute question | Find and cite the right section | 1+ citation and keywords |
| T2 Judgment question | Answer from a court judgment and cite it | Cite the case id |
| T3 Multi-turn | Use a fact from an earlier turn | 2+ turns; a gold turn with `memory_dependent: true` |
| T4 Amendment history | Say which law amended a section and when, from its official amendment notes, where the section was named in an earlier turn | 2+ turns; a gold turn with `memory_dependent: true`; the amendment year among `answer_keywords` |
| T5 Should refuse | The documents cannot answer it, or provisions conflict | Every gold turn has `abstain: true` |

### Writing good memory questions (T3, T4)

The baseline bot remembers only the **last 2 turns**. To test memory properly, put the
key fact at least 3 turns before the question that needs it, with other turns in between.

### Check your file

```bash
python -m harness.validate_sessions contrib/G07.jsonl --contrib
```

---

## 3. Peer review file (`review/<your group>.csv`)

```csv
session_id,decision,reason
G03-T1-01,accept,citation and keywords are correct
G03-T3-02,fix,turn 4 does not really need turn 1; the section is named in turn 4 itself
G03-T5-01,reject,section 210 exists in the corpus so the question is answerable
```

`decision` is one of `accept`, `fix`, `reject`. Give a one-line reason for every row.

---

## 4. The chatbot's answer

Your chatbot must reply with one JSON object:

```json
{"answer": "Section 379 of the Pakistan Penal Code, 1860 provides the punishment for theft.",
 "citations": ["PPC-379"],
 "abstain": false}
```

The harness parses this for you (`harness/answer.py`). If the model replies with
something that is not valid JSON, the answer is still recorded but marked `parse_ok = 0`
(error code E8). Every answer shown to a user must include the disclaimer, which
`Answer.display()` adds.

---

## 5. Results file (`results/<group>_<config>_<split>_s<seed>.csv`)

Written by `python -m harness.evaluate`. One row per marked turn. Never edit it.

| Column | Meaning |
| --- | --- |
| `group_id`, `config`, `memory`, `adapter`, `seed`, `split` | Which run this is, e.g. `G07`, `M2+A1`, `M2`, `A1`, `13`, `dev` |
| `session_id`, `turn`, `family` | Which question (family = type T1 to T5) |
| `correct` | 1 if correct. Keyword rule on the practice set; on the secret test the instructor's fixed judge also checks it |
| `cite_precision`, `cite_recall` | Share of your citations that are right, and share of gold citations you gave (empty for refusal turns) |
| `abstain_expected`, `abstained` | Should it have said "I am not sure", and did it |
| `memory_recall` | For memory-dependent turns only: 1 if answered correctly |
| `ungrounded_cites` | Citations of documents the bot never saw this turn, or that do not exist |
| `parse_ok` | 0 if the reply was not valid JSON |
| `tokens_in`, `tokens_out`, `latency_ms`, `mem_tokens` | Cost of the turn |
| `failure_code` | Empty. Copy your E1 to E8 labels into `ERRORS.csv`, not here |

Next to every `.csv` the evaluator writes `.pred.jsonl` (every answer) and `.json`
(the run manifest with config, prompt, adapter and corpus hashes, and summary scores).

---

## 6. Trace format (`traces/<run_id>.jsonl`)

One JSON line per span. Every turn has a root span `turn` and five child spans:
`memory_read`, `retrieve`, `generate`, `cite_check`, `memory_write`.

```json
{"run_id": "G07_M2+A1_dev_s13_20261201-101500", "trace_id": "9f2c...", "span_id": "a1b2...",
 "parent_id": "77de...", "name": "memory_read", "start_ms": 1764570900000, "end_ms": 1764570900012,
 "latency_ms": 12, "attributes": {"memory": "M2", "mem_tokens": 41, "items_read": 3, "items": [...], "text": "..."}}
```

The `turn` span carries `run_id, group_id, config, memory, adapter, seed, split,
session_id, turn, config_hash, prompt_hash, adapter_hash, tokens_in, tokens_out,
latency_ms, mem_tokens` and the final answer. Bundle and check them with
`python -m harness.trace_export --group G07`.

---

## 7. Error codes (`ERRORS.csv`, Step 4)

```csv
config,seed,session_id,turn,code,secondary_code,note
M2+A1,13,DEV-T3-0004,4,E3,,fact from turn 1 was never stored
M0+A0,13,DEV-T4-0002,2,E2,E5,used the 2017 version for a 2012 date
```

| Code | Mistake |
| --- | --- |
| E1 | Did not find the right document |
| E2 | Used an old or not-yet-valid version of the law |
| E3 | Forgot a fact from an earlier message |
| E4 | Used a wrong or outdated fact from memory |
| E5 | Cited a section or case that does not support the answer, or does not exist |
| E6 | Answered when it should have said "I am not sure" |
| E7 | Said "I am not sure" when it could have answered |
| E8 | Answer in the wrong format |
