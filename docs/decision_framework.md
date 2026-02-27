# Decision-theoretic evaluation framework

## The core idea
You don't deploy a model to maximize accuracy.
You deploy a model to make **decisions** with **consequences**.

Two models can have the same accuracy but radically different business impact
because FP and FN costs differ.

## Cost matrix (binary)
We define per-use-case costs in `configs/decision_costs.yaml`:

- TP: benefit (often negative cost)
- TN: neutral or minor benefit
- FP: cost of false alarm / unnecessary action
- FN: cost of missed detection / incident

Expected cost at threshold t:
- predict positive if p >= t
- compute mean cost across outcomes

## Threshold optimization
For each model:
- Evaluate expected cost across a threshold grid
- Choose threshold that minimizes expected cost

This often yields:
- A *different* threshold than "0.5"
- A *different* winning model than accuracy-based selection

## Slice-specific decisions
We repeat the same expected cost computation per slice.
Result:
- One model can be best globally but unacceptable in a critical segment.

## Output: defensible recommendation
The recommendation includes:
- Winning model and threshold
- Ranking with expected cost values
- Per-slice recommendations where relevant
- Instability flags to prevent overconfident decisions on tiny slices
