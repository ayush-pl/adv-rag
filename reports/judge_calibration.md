# Judge calibration

- Judge model `qwen2.5:7b`. The current rubric is `judge-v2`.
- A human scored each answer blind, without the judge's scores or the system's identity.
- Kappa is quadratic-weighted, with a 95% bootstrap interval. Bias is the judge's mean score minus the human's: negative means the judge is harsher.
- *Correct agreement* is agreement on correct (4 or 5) versus not, which is how correctness is used in the generation report.

- **calibration** set: calibration split, used to design judge-v2, so its agreement is optimistic.
- **validation** set: development split, never used to change the judge.

| Rubric | Set | Dimension | Pairs | Exact | Within 1 | Bias | Weighted kappa | Correct agreement |
|---|---|---|---|---|---|---|---|---|
| judge-v1 | calibration | correctness | 50 | 0.22 | 0.78 | -0.96 | 0.40 [0.19, 0.56] | 76% |
| judge-v1 | calibration | faithfulness | 50 | 0.64 | 0.76 | -0.58 | 0.22 [-0.01, 0.46] |  |
| judge-v1 | calibration | relevance | 50 | 0.52 | 0.60 | -0.98 | 0.24 [0.14, 0.31] |  |
| judge-v1 | validation | correctness | 24 | 0.58 | 0.96 | -0.33 | 0.79 [0.46, 0.93] | 96% |
| judge-v1 | validation | faithfulness | 24 | 0.62 | 0.75 | -0.62 | 0.11 [-0.17, 0.48] |  |
| judge-v1 | validation | relevance | 24 | 0.67 | 0.83 | -0.42 | 0.78 [0.53, 0.91] |  |
| judge-v2 | calibration | correctness | 50 | 0.24 | 0.82 | -0.92 | 0.40 [0.18, 0.57] | 76% |
| judge-v2 | calibration | faithfulness | 50 | 0.66 | 0.82 | -0.46 | 0.31 [0.04, 0.58] |  |
| judge-v2 | calibration | relevance | 50 | 0.72 | 0.78 | -0.56 | 0.34 [0.22, 0.42] |  |
| judge-v2 | validation | correctness | 24 | 0.54 | 0.92 | -0.46 | 0.82 [0.57, 0.92] | 88% |
| judge-v2 | validation | faithfulness | 24 | 0.58 | 0.62 | -0.88 | 0.06 [-0.18, 0.35] |  |
| judge-v2 | validation | relevance | 24 | 0.79 | 0.88 | -0.25 | 0.85 [0.65, 0.96] |  |

## Largest disagreements for judge-v2

Differences of 2 or more points.

| Set | Sample | Dimension | Human | Judge | Judge's reason | Human note |
|---|---|---|---|---|---|---|
| calibration | s-021 | faithfulness | 5 | 1 | The claims in the answer are not supported by the passages. The passages mention that traffic goes through an 802.1Q sub-interface and that this allow | Describes MAC-address routing and network visibility, but omits the 802.1Q sub-i |
| calibration | s-026 | faithfulness | 5 | 1 | The answer does not provide any supporting details from the passages, and it does not mention the `immutable: true` field which is crucial for creatin |  |
| validation | v-007 | faithfulness | 5 | 1 | The passage does not provide any information about the types of indexes that allow the data type stored in the index to differ from the column data ty | Discusses index-only scans rather than which index types support different stora |
| validation | v-014 | faithfulness | 5 | 1 | The answer does not contain any claims that are supported by the passages, as the passages do not discuss loading Docker images from a `.tar` file. | Explains image creation and export, not loading an existing tar archive. |
| validation | v-020 | faithfulness | 5 | 1 | The passage does not provide the specific configuration details for the `synchronous_standby_names` parameter. | Describes candidate standby behavior but fails to provide the required ANY 2 con |
| calibration | s-008 | relevance | 4 | 1 | The answer does not address the question about configuring PostgreSQL to store query results in host variables using a cursor. Instead, it discusses a | Adds a misleading array-size claim: the passage presents arrays as an alternativ |
| calibration | s-021 | relevance | 4 | 1 | The answer does not address the specific question about 802.1Q trunk bridge mode. It instead provides information about bridge mode. | Describes MAC-address routing and network visibility, but omits the 802.1Q sub-i |
| calibration | s-026 | correctness | 5 | 2 | The answer is too brief and does not fully capture the required details. It omits the specific field `immutable: true` that needs to be set in the Con |  |
| calibration | s-038 | relevance | 4 | 1 | The answer does not address the question asked, which is about the difference between creating a new trigger and replacing an existing one. | The first sentence says both statements create new triggers, which conflicts wit |
| calibration | s-042 | relevance | 4 | 1 | The answer does not address the question about configuring Docker to only allow signed releases, which is off-topic. | Describes trustpinning for signed images, not the release-build Git-tag check sp |
| calibration | s-049 | correctness | 4 | 1 | The answer does not address the question asked. It provides information about upgrading Kubernetes and `kubectl`, but does not mention the `kubeadm co | reference wrong — the supplied passages recommend upgrading Kubernetes patch/min |
| calibration | s-049 | relevance | 4 | 1 | The answer does not address the question about the recommended way to ensure a Kubernetes configuration is up-to-date with the latest version. Instead | reference wrong — the supplied passages recommend upgrading Kubernetes patch/min |
| calibration | s-050 | correctness | 5 | 2 | The answer is too brief and does not fully capture the nuance of the passage. |  |
| calibration | s-050 | faithfulness | 5 | 2 | The answer does not mention the condition that if only one port is specified, it applies to all the hosts. |  |
| validation | v-002 | faithfulness | 5 | 2 | The answer only covers the PostgreSQL part of the question and does not include the Kubernetes part, which is a significant omission. | Addresses general PostgreSQL CASCADE behavior but does not specifically address  |
| validation | v-016 | faithfulness | 2 | 5 | The claim is directly supported by the passage, which explicitly states this value. | The 300-second value appears in the passage's example configuration, but the ans |
| calibration | s-007 | faithfulness | 5 | 3 | The claim about `kubectl` is supported by the passage, but the statement about the API server handling all requests is not accurate and introduces an  |  |
| calibration | s-008 | correctness | 3 | 1 | The answer does not address the specific question about storing results in host variables using a cursor. It instead discusses arrays and buffer overf | Adds a misleading array-size claim: the passage presents arrays as an alternativ |
| calibration | s-008 | faithfulness | 3 | 1 | The answer does not mention the `FETCH` command with an `INTO` clause, which is a key part of the reference answer and is not supported by the passage | Adds a misleading array-size claim: the passage presents arrays as an alternativ |
| calibration | s-017 | correctness | 5 | 3 | The answer is partially correct but lacks the detail that it can be overridden. It does not mention the possibility of using `NOINHERIT`. |  |
| calibration | s-022 | correctness | 5 | 3 | The answer is mostly correct but lacks the detail about the `verify_git_signature` function and the condition involving `input.git.tagName`. It does n |  |
| calibration | s-022 | faithfulness | 5 | 3 | The answer is not fully faithful to the passage. It omits the specific condition and function that are crucial to the configuration. |  |
| calibration | s-022 | relevance | 5 | 3 | The answer addresses the question by mentioning the `target` environment variable, which is relevant, but it does not fully capture the complete confi |  |
| calibration | s-024 | relevance | 5 | 3 | The answer addresses the question asked, but it provides more information than necessary. |  |
| calibration | s-025 | correctness | 5 | 3 | The answer correctly identifies the return types for PL/pgSQL and PL/Tcl, but it incorrectly states that PL/pgSQL uses `TG_something` instead of `TG_n |  |
| calibration | s-026 | relevance | 5 | 3 | The answer addresses the question but is incomplete and lacks the necessary details to fully answer the question. |  |
| calibration | s-033 | relevance | 5 | 3 | The answer addresses the question asked, but includes an additional detail about other entities that is not relevant to the specific question. |  |
| calibration | s-038 | faithfulness | 3 | 1 | The answer does not support the claim that `CREATE TRIGGER` creates new triggers, as this is not mentioned in the passages. The passages only discuss  | The first sentence says both statements create new triggers, which conflicts wit |
| calibration | s-040 | correctness | 5 | 3 | The answer is correct but does not fully address the question. It mentions both direct interaction and `kubectl`, but the question specifically asks a |  |
| calibration | s-040 | relevance | 5 | 3 | The answer addresses the question by mentioning the interaction method, but it does not fully capture the typical method, which is `kubectl`. |  |

Reference answers flagged as wrong by the reviewer: s-011 (q-f990fbd76f), s-049 (q-b9ca9f7ac0), v-017 (q-c3e3e93bf0).

Left out of every row, because not every rubric version judged them (their reference answer was corrected after scoring): v-017 (validation).

## Known limitations

- One human rater, so human-to-human agreement is not measured.
- Small samples, so intervals are wide and per-category agreement is not reported.
- Most answers were good, so scores cluster at 5. That makes kappa unstable, so read it alongside exact agreement and bias.
- Abstentions and unanswerable questions are scored deterministically, not by the judge, so this calibration does not cover them.
