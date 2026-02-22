# Deep Tree Echo — Custom ReservoirPy Nodes
# AAR Mapping: Reservoir=Arena, Readout=Agent, Relation=Self
"""
dte_nodes: Custom ReservoirPy nodes for the Deep Tree Echo cognitive architecture.

Node hierarchy:
    EchoReservoir (Arena)      — Multi-scale ESN reservoir with fast/slow dynamics
    CognitiveReadout (Agent)   — Trainable ridge readout with online learning
    AARRelation (Self)         — Feedback coupling between agent and arena
    EchobeatNode               — 9-step cognitive cycle orchestrator
    IntrospectionNode          — Recursive self-monitoring with depth control
    MembraneNode               — Hierarchical membrane boundary with gating
"""

from dte_nodes.echo_reservoir import EchoReservoir
from dte_nodes.cognitive_readout import CognitiveReadout
from dte_nodes.aar_relation import AARRelation
from dte_nodes.echobeat_node import EchobeatNode
from dte_nodes.introspection_node import IntrospectionNode
from dte_nodes.membrane_node import MembraneNode

__all__ = [
    "EchoReservoir",
    "CognitiveReadout",
    "AARRelation",
    "EchobeatNode",
    "IntrospectionNode",
    "MembraneNode",
]
