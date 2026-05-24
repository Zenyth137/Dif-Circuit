#!/usr/bin/env python3
"""
Generate synthetic netlist datasets for training and evaluation.

Usage:
    python scripts/generate_netlists.py --num-modules 5000 --num-nets 5000 --num-samples 100 --output data/train
    python scripts/generate_netlists.py --num-modules 2000 --num-nets 2000 --num-samples 20 --output data/test
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.netlist.generator import NetlistGenerator, NetlistConfig


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic VLSI netlist datasets"
    )
    parser.add_argument("--num-modules", type=int, default=5000,
                        help="Number of modules per netlist (default: 5000)")
    parser.add_argument("--num-nets", type=int, default=5000,
                        help="Number of nets per netlist (default: 5000)")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of netlists to generate (default: 100)")
    parser.add_argument("--canvas-width", type=float, default=1000.0)
    parser.add_argument("--canvas-height", type=float, default=1000.0)
    parser.add_argument("--min-width", type=float, default=2.0)
    parser.add_argument("--max-width", type=float, default=50.0)
    parser.add_argument("--min-height", type=float, default=2.0)
    parser.add_argument("--max-height", type=float, default=50.0)
    parser.add_argument("--min-pins", type=int, default=2)
    parser.add_argument("--max-pins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/netlists",
                        help="Output directory")
    parser.add_argument("--prefix", type=str, default="netlist",
                        help="Filename prefix")

    args = parser.parse_args()

    config = NetlistConfig(
        num_modules=args.num_modules,
        num_nets=args.num_nets,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
        min_pins_per_net=args.min_pins,
        max_pins_per_net=args.max_pins,
        seed=args.seed,
    )

    generator = NetlistGenerator(config)
    print(f"Generating {args.num_samples} netlists with "
          f"{args.num_modules} modules, {args.num_nets} nets each...")

    for i in range(args.num_samples):
        # Use different seed for each netlist
        config.seed = args.seed + i
        generator.config = config
        nodes, nets = generator.generate()
        generator.save(args.output, f"{args.prefix}_{i:04d}")

        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{args.num_samples} netlists")

    print(f"Done! Saved to {args.output}/")


if __name__ == "__main__":
    main()
