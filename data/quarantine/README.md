# Quarantine — sources requiring a human licensing decision

A source lands here when its license does **not** clearly permit *both* ML-training
use *and* redistribution of a **derived** dataset. Per the project's HARD
CONSTRAINT, these are never merged into the training set without an explicit
human decision (and, where noted, written permission / relicensing from the
rights holder).

Each entry records: the source, the governing license, the verbatim clause, the
URL it came from, and why it is blocked. Full machine-readable findings for every
source (including the usable ones) live in
[`configs/license_inventory.json`](../../configs/license_inventory.json).

| Source | Tier | License (of the DATA) | Blocker | Verbatim clause (source) |
| --- | --- | --- | --- | --- |
| **PMechDB** | A | CC-BY-NC-ND-4.0 | **ND** forbids redistributing a derived/reformatted corpus; **NC** forbids commercial use | "The data is released under the CC-BY-NC-ND license… distribute the **unmodified** data (with proper reference), but **may not use it commercially**." — https://deeprxn.ics.uci.edu/pmechdb/download |
| **RMechDB** | A | CC-BY-NC-ND-4.0 | Same ND + NC blockers | "released under the CC-BY-NC-ND license… distribute it as long as they reference the original work (RMechDB)… **not allowed to change the data in any way or use them commercially**." — https://deeprxn.ics.uci.edu/rmechdb/download |
| **OpenStax Organic Chemistry** | C | CC-BY-NC-SA-4.0 + explicit no-AI clause | Publisher explicitly **prohibits LLM training / ingestion** without permission; NC | OpenStax terms: no ingestion/training of AI without OpenStax permission (see openstax.org terms). |

## Why this matters (spec correction)

The task spec optimistically listed PMechDB as "Public / academic" and RMechDB as
"CC-BY 4.0". **Both are wrong for the dataset itself.** The associated *journal
articles* are CC-BY-4.0, but the *dataset download pages* place the data under
**CC-BY-NC-ND-4.0** — verified by direct fetch of both download pages (2026-07-09).

Consequence: the spec's **primary Tier-A source of typed polar + radical
elementary steps is not ingestible** for a redistributable dataset under current
terms. This is a material change to the data plan and needs a human decision
(see the Phase 0 gate).

## To clear an entry

1. Obtain written permission / a relicense from the rights holder
   (PMechDB/RMechDB: Baldi Research Group, UC Irvine — pfbaldi@uci.edu), **or**
2. Decide to build a **non-commercial, non-redistributed, internal-research-only**
   variant of the dataset and record that scope decision here, **or**
3. Drop the source and proceed with the licensed alternatives (oMe-Silver +
   template expansion for Tier A; ORD + USPTO for Tier B; Wikipedia for Tier C).

Until then, do not run these ingesters against a released build.
