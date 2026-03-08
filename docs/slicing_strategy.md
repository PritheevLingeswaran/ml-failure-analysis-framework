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

## Metric validity on slices
Some positive-class metrics are undefined on single-class slices.
The framework now makes that explicit instead of silently coercing them to zero.

- If a slice has no positive labels, positive-class recall is `null` and cannot trigger a metric-drop diagnostic.
- If a slice has no predicted positives at the evaluated threshold, precision is `null` and cannot trigger a metric-drop diagnostic.
- `f1` is `null` whenever its precision/recall inputs are undefined.
- Slice outputs include `metrics_validity` with boolean validity flags and reasons such as `no_positive_labels`, `no_predicted_positives`, and `single_class_slice`.

This matters because slices like `label == 0` can otherwise create bogus "recall dropped to zero" findings even though positive recall is not a meaningful quantity on that slice.
