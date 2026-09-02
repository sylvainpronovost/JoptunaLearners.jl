# Architecture ledger

> Evidence status: historical results and revision labels below are inherited, not a current Joptuna hosted qualification. See the repository-root `QUALIFICATION.md`.

“Structural qualification” means the native Lux graph implements the documented computational
pieces and its model-specific invariants are executable and differentiable. It does not claim
seed-identical behavior with another framework.

| Group | Models | Contract surface | Current evidence |
|---|---|---|---|
| Tabular | MLPRegressor, TabularMixer, TabularResNet | Native Lux, point output | construction, gradients, updates, replay, restoration, serialization, and memory lifecycle |
| Window-linear | WindowMLP, WindowLinear, WindowNLinear, WindowDLinear | Native Lux, causal window input | structural and supervised backend lifecycle qualification |
| Temporal | TSMixer, FiLMTSMixer, TCN, TCNV2 | RevIN; causal dilated residual blocks; nonlinear input adapter | structural qualification and backend lifecycle tests |
| Advanced | PatchTSTLite | channel-mixing patches, positional parameters, pre-norm attention encoder, GELU feed-forward blocks | structural and backend lifecycle tests |
| Advanced | TFTLite | feature GRN, stacked LSTM encoder, last-step attention, post-attention GRN | structural qualification and backend lifecycle tests |
| Advanced | MambaLite | pre-norm gated projection, causal depthwise convolution, learned-decay state scan, residual output block | structural qualification and backend lifecycle tests |

The registry mechanically preserves current names, defaults, and conditional search schemas.
Dropout configuration is retained in the public contract. No claim of foreign checkpoint
compatibility or seed-identical tensors is made.
