# NAS Optimization Report

## Identified problem
Binary Classification

## Data modality
1D Time-series (ECG)

## Chosen strategy
Agentic NAS with profiler -> builder -> trainer (Optuna) -> evaluator, using bounded search spaces and ranking models by objective score plus cost/quality trade-off.

## Architecture families considered
- `cnn1d`: best_value=0.738237524921743, quality=moderate, efficiency=medium
- `rnn`: best_value=0, quality=weak, efficiency=medium

## Best configuration found
- Selected family: `cnn1d`
- `dropout`: `0.3`
- `filters`: `32`
- `kernel_size`: `11`
- `num_conv_layers`: `3`
- `pooling_type`: `max`
- `stride`: `1`

## Final performance
- Best family: `cnn1d`
- Best objective value: `0.738237524921743`
- Margin vs next candidate: `0.738238`

## Justification for the choice
- Highest observed objective value (0.7382).
- Efficiency tier is medium with cost score 4.7.
- Margin vs second option (rnn): 0.7382.
- Trade-off summary: Chosen family balances score 0.7382 with medium estimated cost.

## Limitations
- Some instability/complexity signals exist; monitor variance in additional runs.

## Next steps
- Adopt 'cnn1d' as current best candidate with params: {'filters': 32, 'pooling_type': 'max', 'stride': 1, 'num_conv_layers': 3, 'kernel_size': 11, 'dropout': 0.3}.
- Run a confirmatory experiment with more trials and fixed seed for reproducibility.
- Track accuracy versus compute cost in the next iteration to validate deployment fit.
