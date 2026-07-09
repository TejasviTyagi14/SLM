# Rejects — records that failed a pipeline gate, with reason codes

Records here were dropped by a validation gate and are NEVER in the final set.
Each file is JSONL; every row carries the source, reaction_id, and a reason code.

## Phase 3 — `phase3_unmapped.jsonl`

A mechanism step's `(type, subtype)` could not be mapped to the benchmark's
allowed ontology (`configs/ontology.json`).

| Reason code | Meaning |
| --- | --- |
| `unmapped_step_type:<type>::<subtype>` | Pair is neither in the allowed ontology nor in the `template_remap` table. |
| `unmapped_step_type:<...>(explicit_reject)` | Pair is in `template_remap` but mapped to `null` — a genuinely ambiguous/non-ontological label (e.g. `rearrangement::rearrangement`, `aromatization::dehydration`). Rejected rather than guessed, per the "no fabricated chemistry" rule. |
| `remap_target_not_allowed:<...>` | A remap entry points at a pair not in the allowed set (should never happen; indicates an ontology.json bug). |

Note: some Silver records are rejected because their *parent template* (e.g.
NR-202) carries an ambiguous label. We reject rather than remap-by-guess, because
choosing a concrete mechanism type for an ambiguous label would fabricate
chemistry. These are recoverable only by an expert relabel of the source template.

## Phase 5 — (added when Phase 5 lands)

Sanitization / valence / atom-balance / charge-balance / step-continuity /
no-op-step failures, each with its own reason code.
