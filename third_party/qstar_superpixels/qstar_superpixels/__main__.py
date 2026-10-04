"""Run on a tensor batch or a single RGB image; save all hierarchy levels."""

import argparse
from pathlib import Path

import torch

from . import PipelineConfig, SuperpixelPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="RGB image or .pt float tensor [B, C, H, W]")
    parser.add_argument("output", type=Path, help="Output .pt file")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--checkpoint", type=Path, help="Notebook tierfit state_dict; enables learned features")
    parser.add_argument("--target-k", type=int, default=512)
    parser.add_argument("--max-level", type=int, default=5)
    args = parser.parse_args()
    if args.input.suffix.lower() == ".pt":
        x = torch.load(args.input, map_location="cpu", weights_only=True)
    else:
        from PIL import Image
        with Image.open(args.input) as image:
            image = image.convert("RGB")
            x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).float()
            x = x.reshape(image.height, image.width, 3).permute(2, 0, 1)[None] / 255
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        x = (x - mean) / std
    config = PipelineConfig(target_k=args.target_k, max_level=args.max_level)
    model = SuperpixelPipeline(config, learned=args.checkpoint is not None)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(args.device).eval()
    with torch.no_grad():
        result = model(x.to(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "segmentations": tuple(s.cpu() for s in result.all_segs()),
        "mappings": tuple(m.cpu() for m in result.mappings),
        "features": tuple(f.cpu() for f in result.features),
        "batch_indices": tuple(b.cpu() for b in result.batch_indices),
    }, args.output)
    print("Regions per image at each level:", [c.tolist() for c in result.region_counts()])


if __name__ == "__main__":
    main()
