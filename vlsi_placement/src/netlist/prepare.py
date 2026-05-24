"""
Helpers to cap netlist size and build graphs for GPU-bound training/inference.
"""

from typing import Tuple, List, Optional
import numpy as np

from .generator import NetlistGenerator, NetlistConfig


def truncate_netlist(
    nodes: np.ndarray,
    nets: list,
    max_modules: int,
    min_pins_per_net: int = 2,
) -> Tuple[np.ndarray, list]:
    """Keep first ``max_modules`` modules and nets whose pins lie in range."""
    if max_modules is None or max_modules <= 0:
        return nodes, nets

    max_modules = min(max_modules, len(nodes))
    nodes_out = nodes[:max_modules].copy()
    nodes_out[:, 0] = np.arange(max_modules)

    filtered_nets: List[list] = []
    for net in nets:
        pins = [m for m in net if m < max_modules]
        if len(pins) >= min_pins_per_net:
            filtered_nets.append(sorted(pins))

    return nodes_out, filtered_nets


def attach_netlist_to_generator(
    generator: NetlistGenerator,
    nodes: np.ndarray,
    nets: list,
) -> NetlistGenerator:
    """Set generator internal state after loading or truncating a netlist."""
    generator._nodes = nodes
    generator._nets = nets
    generator.config.num_modules = len(nodes)
    return generator


def prepare_netlist_graph(
    nodes: np.ndarray,
    nets: list,
    max_modules: Optional[int] = None,
    min_pins_per_net: int = 2,
    bipartite: bool = False,
) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    """
    Optionally truncate, then build edge_index / edge_attr.

    Args:
        bipartite: if True, module↔net edges (PolicyNet); else module↔module (DenoiserNet)
    """
    if max_modules is not None and max_modules > 0:
        nodes, nets = truncate_netlist(nodes, nets, max_modules, min_pins_per_net)

    gen = NetlistGenerator(NetlistConfig(num_modules=len(nodes), num_nets=len(nets)))
    attach_netlist_to_generator(gen, nodes, nets)

    if bipartite:
        edge_index, edge_attr = gen.build_edge_index()
    else:
        edge_index, edge_attr = gen.build_module_edge_index()

    return nodes, nets, edge_index, edge_attr
