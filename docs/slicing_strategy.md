# Slicing strategy

## Why slicing exists
If you rely on overall metrics, you are choosing to be blind to:
- Segment-specific failure rates
- Simpson's paradox (global improves while a critical segment degrades)
- Incident drivers (rare but expensive segments)

## Slice types supported

### 1) Rule-based slices (`configs/slices.yaml`)
- Defined as pandas query expressions.
- Reviewed like code because slice definitions *change decisions*.
- If the dataset is missing required columns, the slice is skipped (logged).

Examples:
- `label == 1`
- `region == 'US'`
- `amount >= 1000`

### 2) Automatic slices
Configured in `configs/base.yaml`:
- **Text length bins**: proxies UX/log verbosity or request complexity
- **Confidence bins**: exposes overconfidence and calibration problems
- **Label frequency bins**: detects failures on rare labels
- **Easy vs hard**: simple baseline uses ensemble disagreement; disagreement ≈ ambiguous/hard

## Slice instability
Small slices lie easily. This framework flags:
- Low sample count slices (configurable min_count)
- (Extendable) CI-based instability via bootstrap

Treat unstable slices as **hypothesis generators**, not decision drivers.
