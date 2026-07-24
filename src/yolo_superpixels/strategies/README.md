# Strategies

Each strategy is one way of exposing superpixel information to YOLO.

- `input_channels`: RGB input is kept externally, while
  the first convolution internally receives RGB plus two adjacency channels.
- `neck_injection`: experimental implementation that injects SPiT or lightweight
  region tokens into the YOLO neck at the P3 Detect input.
- `auxiliary_loss`: training-only auxiliary head that asks an intermediate YOLO
  feature map to reproduce superpixel right/down structure. The head weights are
  saved in checkpoints, but the block is inactive in eval/predict mode.

New strategy checklist:

1. Create `src/yolo_superpixels/strategies/<strategy_name>/`.
2. Implement a strategy adapter with `patch_model()` and `prepare_training()`.
3. Put any framework-specific trainer code under `src/yolo_superpixels/integrations/`.
4. Add a small example under `examples/` and a config under `configs/`.
5. Add focused tests under `tests/`.
