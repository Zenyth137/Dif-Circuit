"""
Netlist parser for reading saved netlist CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


class NetlistParser:
    """Parse netlist files saved by NetlistGenerator."""

    @staticmethod
    def load(output_dir: str, prefix: str = "netlist") -> Tuple[np.ndarray, list]:
        """
        Load a netlist from CSV files.

        Args:
            output_dir: Directory containing the CSV files
            prefix: Filename prefix (e.g. 'netlist' → 'netlist_nodes.csv')

        Returns:
            nodes: (N, 3) array [id, width, height]
            nets: list of lists of module indices
        """
        out = Path(output_dir)
        nodes_path = out / f"{prefix}_nodes.csv"
        nets_path = out / f"{prefix}_nets.csv"

        if not nodes_path.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
        if not nets_path.exists():
            raise FileNotFoundError(f"Nets file not found: {nets_path}")

        nodes_df = pd.read_csv(nodes_path)
        nodes = nodes_df[["id", "width", "height"]].values.astype(np.float32)

        nets_df = pd.read_csv(nets_path)
        nets = []
        for _, row in nets_df.iterrows():
            modules = [int(x) for x in str(row["modules"]).split(",")]
            nets.append(modules)

        print(f"Loaded {len(nodes)} nodes, {len(nets)} nets from {output_dir}/")
        return nodes, nets
