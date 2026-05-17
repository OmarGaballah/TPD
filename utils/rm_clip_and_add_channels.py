import argparse
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Expand UNet input/output channels and strip cond_stage_model weights."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the source checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the modified checkpoint (.ckpt)"
    )
    args = parser.parse_args()

    ckpt_file = torch.load(args.input, map_location="cpu")

    # add input conv mask channel
    new_input_weight = torch.zeros(320, 1, 3, 3)
    ckpt_file["state_dict"]["model.diffusion_model.input_blocks.0.0.weight"] = torch.cat((
        torch.cat((
            ckpt_file["state_dict"]["model.diffusion_model.input_blocks.0.0.weight"][:, :4],
            new_input_weight,
        ), dim=1),
        ckpt_file["state_dict"]["model.diffusion_model.input_blocks.0.0.weight"][:, 4:],
    ), dim=1)

    # add input conv pose channel
    new_input_weight = torch.zeros(320, 8, 3, 3)
    ckpt_file["state_dict"]["model.diffusion_model.input_blocks.0.0.weight"] = torch.cat((
        ckpt_file["state_dict"]["model.diffusion_model.input_blocks.0.0.weight"],
        new_input_weight,
    ), dim=1)

    # add output conv mask channel
    new_output_weight = torch.zeros(1, 320, 3, 3)
    ckpt_file["state_dict"]["model.diffusion_model.out.2.weight"] = torch.cat((
        ckpt_file["state_dict"]["model.diffusion_model.out.2.weight"],
        new_output_weight,
    ), dim=0)
    new_output_bias = torch.zeros(1)
    ckpt_file["state_dict"]["model.diffusion_model.out.2.bias"] = torch.cat((
        ckpt_file["state_dict"]["model.diffusion_model.out.2.bias"],
        new_output_bias,
    ), dim=0)

    # strip cond_stage_model weights
    ckpt_file["state_dict"] = {
        k: v for k, v in ckpt_file["state_dict"].items()
        if not k.startswith("cond_stage_model")
    }

    torch.save(ckpt_file, args.output)
    print(f"Saved modified checkpoint to {args.output}")


if __name__ == "__main__":
    main()
