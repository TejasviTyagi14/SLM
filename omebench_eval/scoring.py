"""oMeS scoring for oMeBench.

This is a faithful re-implementation of the official oMeBench `oMeS` metric
(weighted Needleman-Wunsch alignment with Morgan-fingerprint Tanimoto
similarity). Keeping the algorithm identical means scores produced here are
directly comparable to the numbers reported in the oMeBench paper.

Metrics returned per reaction:
- V        : fraction of predicted intermediates that are valid SMILES
- L        : logical fidelity = (# aligned "match" steps) / (# gold steps)
- S_total  : strict score  (exact type + exact canonical structure match)
- S_partial: partial score (type match + similarity-weighted structure credit)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Similarity threshold below which partial structural credit is zeroed out.
TAU: float = 0.60
FP_RADIUS: int = 2
FP_NBITS: int = 2048
# Tiny negative bias so that, all else equal, the aligner prefers real matches
# over gap moves without meaningfully changing the reported score.
NON_MATCH_PENALTY: float = 1e-6

_morgan_gen = GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)

# gold step: (subtype, intermediate_smiles, step_weight)
GoldStep = Tuple[str, str, float]
# predicted step: (subtype, intermediate_smiles)
PredStep = Tuple[str, str]


def canonical_smiles(smi: str) -> str:
    """Canonicalize a SMILES string; return "" if RDKit cannot parse it."""
    if not isinstance(smi, str) or not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    return "" if mol is None else Chem.MolToSmiles(mol, canonical=True)


def tanimoto_morgan(smi1: str, smi2: str) -> float:
    """Tanimoto similarity between two SMILES via Morgan fingerprints."""
    mol1, mol2 = Chem.MolFromSmiles(smi1 or ""), Chem.MolFromSmiles(smi2 or "")
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = _morgan_gen.GetFingerprint(mol1)
    fp2 = _morgan_gen.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def step_sigma(raw_sim: float, tau: float = TAU) -> float:
    return 0.0 if raw_sim < tau else raw_sim


@dataclass
class oMeSResult:
    S_total: float
    S_partial: float
    alignment: List[str]
    V: float
    L: float


def oMeS(
    gold: List[GoldStep],
    pred: List[PredStep],
    sim_fn: Callable[[str, str], float] = tanimoto_morgan,
    tau: float = TAU,
) -> oMeSResult:
    """Align predicted mechanism steps to gold steps and score them."""
    N, M = len(gold), len(pred)

    DP_tot = [[0.0] * (M + 1) for _ in range(N + 1)]
    DP_par = [[0.0] * (M + 1) for _ in range(N + 1)]
    DP_pen = [[0.0] * (M + 1) for _ in range(N + 1)]
    DP_rnk = [[0] * (M + 1) for _ in range(N + 1)]
    trace = [[None] * (M + 1) for _ in range(N + 1)]

    rank = {"match": 3, "type_mismatch": 2, "skip_gold": 1, "skip_pred": 1}
    trace[0][0] = "end"

    for i in range(1, N + 1):
        DP_tot[i][0] = DP_tot[i - 1][0]
        DP_par[i][0] = DP_par[i - 1][0]
        DP_pen[i][0] = DP_pen[i - 1][0] - NON_MATCH_PENALTY
        DP_rnk[i][0] = rank["skip_gold"]
        trace[i][0] = "skip_gold"

    for j in range(1, M + 1):
        DP_tot[0][j] = DP_tot[0][j - 1]
        DP_par[0][j] = DP_par[0][j - 1]
        DP_pen[0][j] = DP_pen[0][j - 1] - NON_MATCH_PENALTY
        DP_rnk[0][j] = rank["skip_pred"]
        trace[0][j] = "skip_pred"

    gold_can = [canonical_smiles(s) for _, s, _ in gold]
    pred_can = [canonical_smiles(s) for _, s in pred]

    for i in range(1, N + 1):
        g_type, g_smi, w = gold[i - 1]
        g_can = gold_can[i - 1]

        for j in range(1, M + 1):
            p_type, p_smi = pred[j - 1]
            p_can = pred_can[j - 1]

            if g_type == p_type:
                m_tot = w if (g_can == p_can and g_can) else 0.0
                m_par = (
                    w * step_sigma(sim_fn(g_smi, p_smi), tau)
                    if (g_can and p_can)
                    else 0.0
                )
                action_diag = "match"
                pen_diag = 0.0
                r_diag = rank["match"]
            else:
                m_tot = m_par = 0.0
                action_diag = "type_mismatch"
                pen_diag = -NON_MATCH_PENALTY
                r_diag = rank["type_mismatch"]

            candidates = [
                (
                    DP_tot[i - 1][j],
                    DP_par[i - 1][j],
                    DP_pen[i - 1][j] - NON_MATCH_PENALTY,
                    rank["skip_gold"],
                    "skip_gold",
                    (i - 1, j),
                ),
                (
                    DP_tot[i][j - 1],
                    DP_par[i][j - 1],
                    DP_pen[i][j - 1] - NON_MATCH_PENALTY,
                    rank["skip_pred"],
                    "skip_pred",
                    (i, j - 1),
                ),
                (
                    DP_tot[i - 1][j - 1] + m_tot,
                    DP_par[i - 1][j - 1] + m_par,
                    DP_pen[i - 1][j - 1] + pen_diag,
                    r_diag,
                    action_diag,
                    (i - 1, j - 1),
                ),
            ]

            best = max(candidates, key=lambda x: (x[0], x[1], x[3], x[2]))
            DP_tot[i][j], DP_par[i][j], DP_pen[i][j], DP_rnk[i][j], trace[i][j], _ = best

    align: List[str] = []
    i, j = N, M
    while not (i == 0 and j == 0):
        if i == 0:
            align.append("skip_pred")
            j -= 1
            continue
        if j == 0:
            align.append("skip_gold")
            i -= 1
            continue

        act = trace[i][j]
        if act in ("match", "type_mismatch"):
            align.append(act)
            i, j = i - 1, j - 1
        elif act == "skip_gold":
            align.append(act)
            i -= 1
        elif act == "skip_pred":
            align.append(act)
            j -= 1
        elif act == "end":
            break
        else:
            raise RuntimeError(f"Unexpected trace action: {act} at ({i}, {j})")
    align.reverse()

    V = sum(1 for _, s in pred if canonical_smiles(s)) / max(1, M)
    L = sum(1 for act in align if act == "match") / max(1, N)
    return oMeSResult(
        round(DP_tot[N][M], 2),
        round(DP_par[N][M], 2),
        align,
        round(V, 2),
        round(L, 2),
    )
