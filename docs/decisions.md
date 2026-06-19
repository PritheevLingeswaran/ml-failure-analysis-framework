# Design choices (and why)

## Why YAML configs
Because evaluation decisions must be reviewable artifacts.
If the config changes, the recommended model can change. That must be explicit.

## Why scikit-learn baselines first
- Cheap, stable, easy to explain
- Provides a minimum viable comparison standard
- Helps detect data leakage and slice issues early

## Why TF-IDF clustering for errors
It's not the fanciest method, but it:
- Finds recurring n-grams
- Is interpretable
- Is deterministic and fast

## Why we treat AUC carefully
AUC can improve while thresholded decision quality worsens.
AUC is useful for ranking, not for selecting an operating point.

## Why slice instability flags
Small slices can flip rankings randomly.
We avoid false confidence by labeling instability early.
