import argparse
import inspect
from pathlib import Path

import onnx
import torch
from onnxsim import simplify

import models.FIQA_model as FIQA_model


def load_state_dict(weights_file):
    checkpoint = torch.load(weights_file, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


def export_model(model, dummy_input, output_file, opset, dynamic_axes=None):
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["quality_score"],
        dynamic_axes=dynamic_axes,
    )


def simplify_onnx(input_file, output_file, input_shape):
    kwargs = {
        "test_input_shapes": {"input": input_shape},
    }
    if "perform_optimization" in inspect.signature(simplify).parameters:
        kwargs["perform_optimization"] = True

    simplified_model, check = simplify(str(input_file), **kwargs)
    if not check:
        raise RuntimeError(f"onnxsim validation failed for {input_file}")

    onnx.save(simplified_model, output_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export FIQA_EdgeNeXt_XXS to fixed-size and dynamic H/W ONNX files."
    )
    parser.add_argument(
        "--model_weights_file",
        type=str,
        default="ckpts/EdgeNeXt_XXS_checkpoint.pt",
        help="Path to FIQA_EdgeNeXt_XXS weights.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="onnx",
        help="Directory for exported ONNX files.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="FIQA_EdgeNeXt_XXS",
        help="Prefix for exported ONNX filenames.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=352,
        help="Square dummy input size used for export and simplification.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = FIQA_model.FIQA_EdgeNeXt_XXS(is_pretrained=False)
    model.load_state_dict(load_state_dict(args.model_weights_file))
    model.eval()

    dummy_input = torch.randn(1, 3, args.image_size, args.image_size)
    fixed_shape = [1, 3, args.image_size, args.image_size]

    fixed_file = output_dir / (
        f"{args.output_prefix}_{args.image_size}x{args.image_size}_opset{args.opset}.onnx"
    )
    fixed_sim_file = output_dir / (
        f"{args.output_prefix}_{args.image_size}x{args.image_size}_opset{args.opset}_sim.onnx"
    )
    dynamic_file = output_dir / (
        f"{args.output_prefix}_dynamic_hw_opset{args.opset}.onnx"
    )
    dynamic_sim_file = output_dir / (
        f"{args.output_prefix}_dynamic_hw_opset{args.opset}_sim.onnx"
    )

    dynamic_hw_axes = {
        "input": {2: "H", 3: "W"},
    }

    print(f"Exporting fixed-resolution ONNX: {fixed_file}")
    export_model(
        model=model,
        dummy_input=dummy_input,
        output_file=fixed_file,
        opset=args.opset,
        dynamic_axes=None,
    )
    print(f"Simplifying fixed-resolution ONNX: {fixed_sim_file}")
    simplify_onnx(fixed_file, fixed_sim_file, fixed_shape)

    print(f"Exporting dynamic H/W ONNX: {dynamic_file}")
    export_model(
        model=model,
        dummy_input=dummy_input,
        output_file=dynamic_file,
        opset=args.opset,
        dynamic_axes=dynamic_hw_axes,
    )
    print(f"Simplifying dynamic H/W ONNX: {dynamic_sim_file}")
    simplify_onnx(dynamic_file, dynamic_sim_file, fixed_shape)

    print("ONNX export completed.")
    print(f"Fixed ONNX: {fixed_file}")
    print(f"Fixed ONNX simplified: {fixed_sim_file}")
    print(f"Dynamic H/W ONNX: {dynamic_file}")
    print(f"Dynamic H/W ONNX simplified: {dynamic_sim_file}")


if __name__ == "__main__":
    main()
