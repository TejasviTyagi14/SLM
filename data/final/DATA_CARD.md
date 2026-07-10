# Data Card — rxndata mechanism-step dataset

A training-ready dataset for a reaction-mechanism LM, optimized for oMeBench/oMeS. The core unit is a typed, balanced elementary mechanistic step with a valid intermediate SMILES.

## Headline

- Clean mechanism records (post validate+decon+dedup): **2338**
- Train / Val: **2217 / 121** (stratified by dominant subtype + level; 30 strata)
- Test: the official **oMe-Gold** (196 rxns), held out — never trained on.
- Ontology: 11 types / 31 subtypes (parsed from the benchmark).

## Provenance

- template_expanded: 2338

## Sources (clean records)

- oMe-Silver: 2338

## Difficulty levels

- easy: 422
- medium: 1680
- hard: 236

## License inventory

| Source | Verdict |
| --- | --- |
| Open Reaction Database (ORD) — ~2M react | USABLE |
| USPTO (Lowe) — "Chemical reactions from  | USABLE |
| RMechDB (Baldi Research Group, UC Irvine | QUARANTINE |
| Wikipedia named reactions (Tier C) — enc | USABLE |
| OpenStax Organic Chemistry (Tier C) — CC | QUARANTINE |
| PMechDB (Public Database of Elementary P | QUARANTINE |
| Chem LibreTexts (chem.libretexts.org) —  | USABLE_FOR_RESEARCH_ONLY |
| oMeBench (Silver/Gold/Template) — GitHub | USABLE |

No paywalled/copyrighted sources are ingested. NC-ND sources (PMechDB, RMechDB) and the no-AI-clause source (OpenStax) are quarantined; LibreTexts (NC) is research-only and kept separable.

## Decontamination

- Removed as contaminated vs oMe-Gold+Template: **7**
- Post-check gold InChIKey leaks: **0** (0 required).

## Ontology coverage (step subtype counts)

| subtype | steps |
| --- | ---: |
| acid_base_proton_transfer | 3278 |
| nucleophilic_addition | 2090 |
| leaving_group_elimination | 1630 |
| nucleophilic_substitution | 692 |
| 1,2-shift | 263 |
| heterolytic_cleavage | 203 |
| sigmatropic_rearrangement | 201 |
| electrophilic_addition | 189 |
| proton_elimination | 178 |
| oxidative_addition | 177 |
| electrophilic_substitution | 139 |
| single_electron_transfer | 127 |
| coordination | 121 |
| reductive_elimination | 76 |
| cycloaddition | 74 |
| electrocyclization | 71 |
| radical_initiation | 63 |
| radical_coupling | 59 |
| radical_substitution | 52 |
| group_transfer | 39 |
| radical_propagation | 36 |
| homolytic_cleavage | 29 |
| radical_elimination | 27 |
| cycloreversion | 27 |
| radical_rearrangement | 20 |
| dissociation | 11 |
| radical_termination | 7 |
| cheletropic_reaction | 4 |

## Task views (Phase 8)


## Known gaps

- Tier-A elementary-step supply is limited to oMe-Silver + template expansion because PMechDB/RMechDB (the spec's primary sources) are CC-BY-NC-ND and cannot be redistributed as a derived corpus.
- Under-covered subtypes (<5 steps) may score poorly on oMeS; consider targeted expansion or licensed sourcing.
- A small set of curated mechanisms whose step 1 introduces reagents disconnected from the substrate (e.g. Appel/NG-007) are rejected by the continuity gate; recoverable with a reagent-introduction step type.
- Atom mapping (RXNMapper) confidence is modest on exotic charged intermediates; low-confidence maps are flagged, not trusted.

## Reproduce

```
make setup   # venv + pinned deps + clone oMeBench
make all     # phases 1-9
make eval-format-check
make test
```
