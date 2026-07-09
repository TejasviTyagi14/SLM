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

## Phase 5 — `phase5_invalid.jsonl`

Concrete mechanism records that failed a validation gate. Reason codes:

| Reason code | Meaning |
| --- | --- |
| `sanitize_fail:<field/step>` | A molecule/intermediate did not pass RDKit sanitization. |
| `balance_unparseable:step<n>` | Intermediate could not be parsed for balance/continuity. |
| `discontinuous_step:step<n>(heavy_delta=<d>)` | Intermediate n is not reachable from n−1 by a single transformation: the main reacting fragments share too little common substructure (MCS below the provenance floor) and the heavy-atom change matches no joining reactant / leaving fragment. |
| `charge_jump:step<n>(delta=<d>)` | Formal charge changed by more than ±2 between consecutive intermediates. |
| `noop_step:step<n>` | Step's intermediate is identical to the previous one (no-op). |

Continuity is checked with a Maximum-Common-Substructure fraction between
consecutive intermediates (main fragments), with a lower floor for expert-authored
(curated / template_expanded) mechanisms than for `inferred` ones.

**Known recoverable pattern (not corruption):** most surviving Phase-5 rejects
(e.g. the Appel-reaction family NG-007) are curated mechanisms whose *step 1
introduces reagents* (CBr4 + PPh3) as a standalone species with no bond to the
substrate yet. Continuity legitimately flags step 1 as disconnected. These are
rejected under the strict "reject, don't fix silently" rule rather than
special-cased; they are recoverable by adding an explicit reagent-introduction
step type or by seeding continuity with declared reagents. Count is small
(~2.8% of concrete records) and confined to a few named-reaction families.
