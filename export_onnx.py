import argparse
import inspect
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import checker, helper, numpy_helper, shape_inference
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


def get_tensor_shapes(model):
    inferred_model = shape_inference.infer_shapes(model)
    tensor_shapes = {}
    tensors = list(inferred_model.graph.input)
    tensors += list(inferred_model.graph.value_info)
    tensors += list(inferred_model.graph.output)

    for value_info in tensors:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        tensor_shapes[value_info.name] = dims

    return tensor_shapes


def get_constant_values(graph):
    constant_values = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    for node in graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                constant_values[node.output[0]] = numpy_helper.to_array(attr.t)
                break

    return constant_values


def get_int_attribute(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return attr.i
    return default


def get_ints_attribute(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return None


def as_int_list(value):
    if value is None:
        return None
    return np.asarray(value).astype(np.int64).reshape(-1).tolist()


def has_squeeze_axis_zero(node, constant_values):
    if node.op_type != "Squeeze":
        return False

    axes = get_ints_attribute(node, "axes")
    if axes is not None:
        return axes == [0]

    if len(node.input) >= 2:
        axes = as_int_list(constant_values.get(node.input[1]))
        return axes == [0]

    return False


def make_int64_constant(name, values):
    tensor = numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=f"{name}_value")
    return helper.make_node("Constant", inputs=[], outputs=[name], name=f"{name}_const", value=tensor)


class NameGenerator:
    def __init__(self, graph):
        names = set()
        for node in graph.node:
            names.update(node.input)
            names.update(node.output)
            if node.name:
                names.add(node.name)
        for initializer in graph.initializer:
            names.add(initializer.name)
        self.names = names

    def make(self, prefix):
        candidate = prefix
        index = 0
        while candidate in self.names:
            index += 1
            candidate = f"{prefix}_{index}"
        self.names.add(candidate)
        return candidate


def build_decomposed_xca_nodes(reshape_node, transpose_node, split_node, squeeze_nodes, channel_dim, name_generator):
    heads = 4
    if channel_dim % (3 * heads) != 0:
        raise RuntimeError(
            f"Cannot decompose {reshape_node.name}: channel dimension {channel_dim} is not divisible by {3 * heads}."
        )
    head_dim = channel_dim // (3 * heads)
    merged_dim = heads * head_dim
    prefix = reshape_node.name or reshape_node.output[0]

    qkv_input = reshape_node.input[0]
    shape_qkv = name_generator.make(f"{prefix}/decomp_shape_qkv")
    batch_index = name_generator.make(f"{prefix}/decomp_batch_index")
    token_index = name_generator.make(f"{prefix}/decomp_token_index")
    batch_scalar = name_generator.make(f"{prefix}/decomp_batch_scalar")
    token_scalar = name_generator.make(f"{prefix}/decomp_token_scalar")
    axes_zero = name_generator.make(f"{prefix}/decomp_axes_zero")
    batch_dim = name_generator.make(f"{prefix}/decomp_batch_dim")
    token_dim = name_generator.make(f"{prefix}/decomp_token_dim")
    three_dim = name_generator.make(f"{prefix}/decomp_three_dim")
    merged_dim_name = name_generator.make(f"{prefix}/decomp_merged_dim")
    first_shape = name_generator.make(f"{prefix}/decomp_first_shape")
    reshape4 = name_generator.make(f"{prefix}/decomp_reshape4")
    transpose4 = name_generator.make(f"{prefix}/decomp_transpose4")
    split_sizes = name_generator.make(f"{prefix}/decomp_split_sizes")
    split_outputs = [
        name_generator.make(f"{prefix}/decomp_split_{index}")
        for index in range(3)
    ]
    squeezed_outputs = [
        name_generator.make(f"{prefix}/decomp_squeezed_{index}")
        for index in range(3)
    ]
    heads_dim = name_generator.make(f"{prefix}/decomp_heads_dim")
    head_dim_name = name_generator.make(f"{prefix}/decomp_head_dim")
    output_shape = name_generator.make(f"{prefix}/decomp_output_shape")

    nodes = [
        helper.make_node("Shape", [qkv_input], [shape_qkv], name=name_generator.make(f"{prefix}/decomp/Shape")),
        make_int64_constant(batch_index, np.array(0, dtype=np.int64)),
        make_int64_constant(token_index, np.array(1, dtype=np.int64)),
        helper.make_node("Gather", [shape_qkv, batch_index], [batch_scalar], name=name_generator.make(f"{prefix}/decomp/GatherBatch"), axis=0),
        helper.make_node("Gather", [shape_qkv, token_index], [token_scalar], name=name_generator.make(f"{prefix}/decomp/GatherToken"), axis=0),
        make_int64_constant(axes_zero, np.array([0], dtype=np.int64)),
        helper.make_node("Unsqueeze", [batch_scalar, axes_zero], [batch_dim], name=name_generator.make(f"{prefix}/decomp/UnsqueezeBatch")),
        helper.make_node("Unsqueeze", [token_scalar, axes_zero], [token_dim], name=name_generator.make(f"{prefix}/decomp/UnsqueezeToken")),
        make_int64_constant(three_dim, np.array([3], dtype=np.int64)),
        make_int64_constant(merged_dim_name, np.array([merged_dim], dtype=np.int64)),
        helper.make_node("Concat", [batch_dim, token_dim, three_dim, merged_dim_name], [first_shape], name=name_generator.make(f"{prefix}/decomp/ConcatFirstShape"), axis=0),
        helper.make_node("Reshape", [qkv_input, first_shape], [reshape4], name=name_generator.make(f"{prefix}/decomp/Reshape4")),
        helper.make_node("Transpose", [reshape4], [transpose4], name=name_generator.make(f"{prefix}/decomp/Transpose4"), perm=[2, 0, 3, 1]),
        make_int64_constant(split_sizes, np.array([1, 1, 1], dtype=np.int64)),
        helper.make_node("Split", [transpose4, split_sizes], split_outputs, name=name_generator.make(f"{prefix}/decomp/Split4"), axis=0),
        make_int64_constant(heads_dim, np.array([heads], dtype=np.int64)),
        make_int64_constant(head_dim_name, np.array([head_dim], dtype=np.int64)),
        helper.make_node("Concat", [batch_dim, heads_dim, head_dim_name, token_dim], [output_shape], name=name_generator.make(f"{prefix}/decomp/ConcatOutputShape"), axis=0),
    ]

    for index, squeeze_node in enumerate(squeeze_nodes):
        nodes.extend(
            [
                helper.make_node("Squeeze", [split_outputs[index], axes_zero], [squeezed_outputs[index]], name=name_generator.make(f"{prefix}/decomp/Squeeze{index}")),
                helper.make_node("Reshape", [squeezed_outputs[index], output_shape], list(squeeze_node.output), name=name_generator.make(f"{prefix}/decomp/ReshapeOutput{index}")),
            ]
        )

    return nodes


def find_xca_decomposition_pattern(reshape_node, tensor_shapes, consumers, constant_values):
    if reshape_node.op_type != "Reshape" or len(reshape_node.output) != 1:
        return None

    reshape_output_shape = tensor_shapes.get(reshape_node.output[0])
    qkv_input_shape = tensor_shapes.get(reshape_node.input[0])
    if not reshape_output_shape or len(reshape_output_shape) != 5:
        return None
    if not qkv_input_shape or len(qkv_input_shape) != 3:
        return None

    channel_dim = qkv_input_shape[-1]
    if not isinstance(channel_dim, int):
        return None

    reshape_consumers = consumers.get(reshape_node.output[0], [])
    if len(reshape_consumers) != 1:
        return None
    transpose_node = reshape_consumers[0]
    if transpose_node.op_type != "Transpose" or get_ints_attribute(transpose_node, "perm") != [2, 0, 3, 4, 1]:
        return None

    transpose_consumers = consumers.get(transpose_node.output[0], [])
    if len(transpose_consumers) != 1:
        return None
    split_node = transpose_consumers[0]
    if split_node.op_type != "Split" or get_int_attribute(split_node, "axis", 0) != 0 or len(split_node.output) != 3:
        return None

    squeeze_nodes = []
    for split_output in split_node.output:
        split_consumers = consumers.get(split_output, [])
        if len(split_consumers) != 1 or not has_squeeze_axis_zero(split_consumers[0], constant_values):
            return None
        squeeze_nodes.append(split_consumers[0])

    return transpose_node, split_node, squeeze_nodes, channel_dim


def remove_stale_value_info(graph, stale_names):
    kept_value_info = [value_info for value_info in graph.value_info if value_info.name not in stale_names]
    del graph.value_info[:]
    graph.value_info.extend(kept_value_info)


def remove_unused_initializers(graph):
    used_inputs = set()
    for node in graph.node:
        used_inputs.update(node.input)

    kept_initializers = [
        initializer
        for initializer in graph.initializer
        if initializer.name in used_inputs
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)


def validate_no_5d_target_ops(model):
    inferred_model = shape_inference.infer_shapes(model)
    tensor_shapes = get_tensor_shapes(inferred_model)
    offenders = []
    for node in inferred_model.graph.node:
        if node.op_type not in {"Reshape", "Split", "Slice"}:
            continue
        for output in node.output:
            shape = tensor_shapes.get(output)
            if shape and len(shape) >= 5:
                offenders.append(f"{node.op_type} {node.name or output} -> {shape}")

    if offenders:
        formatted = "\n".join(offenders)
        raise RuntimeError(f"5D or higher Reshape/Split/Slice outputs remain after decomposition:\n{formatted}")


def decompose_5d_onnx(input_file, output_file):
    model = onnx.load(input_file)
    tensor_shapes = get_tensor_shapes(model)
    graph = model.graph
    constant_values = get_constant_values(graph)
    consumers = {}
    for node in graph.node:
        for node_input in node.input:
            consumers.setdefault(node_input, []).append(node)

    name_generator = NameGenerator(graph)
    nodes_to_skip = set()
    stale_names = set()
    replacement_count = 0
    new_nodes = []

    for node in list(graph.node):
        if id(node) in nodes_to_skip:
            continue

        pattern = find_xca_decomposition_pattern(node, tensor_shapes, consumers, constant_values)
        if pattern is None:
            new_nodes.append(node)
            continue

        transpose_node, split_node, squeeze_nodes, channel_dim = pattern
        replacement_nodes = build_decomposed_xca_nodes(
            reshape_node=node,
            transpose_node=transpose_node,
            split_node=split_node,
            squeeze_nodes=squeeze_nodes,
            channel_dim=channel_dim,
            name_generator=name_generator,
        )
        new_nodes.extend(replacement_nodes)

        removed_nodes = [node, transpose_node, split_node, *squeeze_nodes]
        nodes_to_skip.update(id(removed_node) for removed_node in removed_nodes)
        for removed_node in [node, transpose_node, split_node]:
            stale_names.update(removed_node.output)
        for squeeze_node in squeeze_nodes:
            stale_names.update(squeeze_node.input[:1])
        replacement_count += 1

    if replacement_count == 0:
        raise RuntimeError(f"No decomposable 5D XCA patterns found in {input_file}")

    del graph.node[:]
    graph.node.extend(new_nodes)
    remove_stale_value_info(graph, stale_names)
    remove_unused_initializers(graph)
    checker.check_model(model)
    validate_no_5d_target_ops(model)
    onnx.save(model, output_file)
    print(f"Applied 5D decomposition to {output_file} ({replacement_count} patterns)")


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
    parser.add_argument(
        "--decomposition",
        action="store_true",
        help="Decompose 5D Reshape/Split/Slice patterns into 4D-compatible ONNX subgraphs.",
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
    if args.decomposition:
        print(f"Applying 5D decomposition: {fixed_sim_file}")
        decompose_5d_onnx(fixed_sim_file, fixed_sim_file)

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
    if args.decomposition:
        print(f"Applying 5D decomposition: {dynamic_sim_file}")
        decompose_5d_onnx(dynamic_sim_file, dynamic_sim_file)

    print("ONNX export completed.")
    print(f"Fixed ONNX: {fixed_file}")
    print(f"Fixed ONNX simplified: {fixed_sim_file}")
    print(f"Dynamic H/W ONNX: {dynamic_file}")
    print(f"Dynamic H/W ONNX simplified: {dynamic_sim_file}")


if __name__ == "__main__":
    main()
