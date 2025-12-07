from typing import List, Tuple, Dict

# Canonical gate set for QCGPT2
# One-qubit standard gates
ONEQ_STD = ["ID", "X", "Y", "Z", "H", "S", "T"]

# One-qubit rotations kept (redundant angles mapped to std gates elsewhere)
ONEQ_RX = ["RX_PI_16", "RX_PI_8", "RX_PI_4", "RX_PI_2"]
ONEQ_RY = ["RY_PI_16", "RY_PI_8", "RY_PI_4", "RY_PI_2"]
ONEQ_RZ = ["RZ_PI_16", "RZ_PI_8"]

TWOQ = ["CX", "CZ", "SWAP"]  # CZ/SWAP symmetric (a<b), CX ordered

THREEQ = ["CCX", "CCZ", "CSWAP"]


def canonicalize_targets(gate_type: str, targets: List[int]) -> List[int]:
    if gate_type in {"CZ", "SWAP"} and len(targets) == 2:
        a, b = targets
        return [min(a, b), max(a, b)]
    if gate_type == "CCZ" and len(targets) == 3:
        return sorted(targets)
    if gate_type == "CSWAP" and len(targets) == 3:
        ctrl, a, b = targets[0], targets[1], targets[2]
        a, b = sorted([a, b])
        return [ctrl, a, b]
    if gate_type == "CCX" and len(targets) == 3:
        c1, c2, t = targets[0], targets[1], targets[2]
        if c1 > c2:
            c1, c2 = c2, c1
        return [c1, c2, t]
    return targets


def build_vocab(n_qubits: int = 3) -> List[str]:
    toks: List[str] = []
    # 1Q std
    for g in ONEQ_STD:
        for q in range(n_qubits):
            toks.append(f"{g}_{q}")
    # 1Q rotations
    for g in ONEQ_RX:
        for q in range(n_qubits):
            toks.append(f"{g}_{q}")
    for g in ONEQ_RY:
        for q in range(n_qubits):
            toks.append(f"{g}_{q}")
    for g in ONEQ_RZ:
        for q in range(n_qubits):
            toks.append(f"{g}_{q}")
    # 2Q
    for a in range(n_qubits):
        for b in range(n_qubits):
            if a == b:
                continue
            toks.append(f"CX_{a}_{b}")
    for a in range(n_qubits):
        for b in range(a + 1, n_qubits):
            toks.append(f"CZ_{a}_{b}")
            toks.append(f"SWAP_{a}_{b}")
    # 3Q
    # CCX: controls symmetric
    for a in range(n_qubits):
        for b in range(a + 1, n_qubits):
            for t in range(n_qubits):
                if t in (a, b):
                    continue
                toks.append(f"CCX_{a}_{b}_{t}")
    # CCZ fully symmetric
    for a in range(n_qubits):
        for b in range(a + 1, n_qubits):
            for c in range(b + 1, n_qubits):
                toks.append(f"CCZ_{a}_{b}_{c}")
    # CSWAP: control + sorted pair
    for ctrl in range(n_qubits):
        for a in range(n_qubits):
            if a == ctrl:
                continue
            for b in range(a + 1, n_qubits):
                if b == ctrl:
                    continue
                toks.append(f"CSWAP_{ctrl}_{a}_{b}")
    return toks


def token_to_gate_parts(tok: str) -> Tuple[str, List[int]]:
    parts = tok.split("_")
    gt = parts[0]
    targets = list(map(int, parts[1:])) if len(parts) > 1 else []
    return gt, canonicalize_targets(gt, targets)


def rotation_to_gate(axis: str, theta: float) -> str:
    import math
    angles = [math.pi/16, math.pi/8, math.pi/4, math.pi/2, math.pi]
    diffs = [abs(theta - a) for a in angles]
    idx = int(diffs.index(min(diffs)))
    nearest = angles[idx]
    eps = 1e-6
    if axis == "rx":
        if abs(nearest - math.pi) < eps:
            return "X"
        if abs(nearest - math.pi/2) < eps:
            return "RX_PI_2"
        if abs(nearest - math.pi/4) < eps:
            return "RX_PI_4"
        if abs(nearest - math.pi/8) < eps:
            return "RX_PI_8"
        return "RX_PI_16"
    if axis == "ry":
        if abs(nearest - math.pi) < eps:
            return "Y"
        if abs(nearest - math.pi/2) < eps:
            return "RY_PI_2"
        if abs(nearest - math.pi/4) < eps:
            return "RY_PI_4"
        if abs(nearest - math.pi/8) < eps:
            return "RY_PI_8"
        return "RY_PI_16"
    if axis == "rz":
        if abs(nearest - math.pi) < eps:
            return "Z"
        if abs(nearest - math.pi/2) < eps:
            return "S"
        if abs(nearest - math.pi/4) < eps:
            return "T"
        if abs(nearest - math.pi/8) < eps:
            return "RZ_PI_8"
        return "RZ_PI_16"
    return "ID"

def apply_to_qiskit(qc, gate_type: str, targets: List[int]):
    import numpy as np
    t = canonicalize_targets(gate_type, targets)
    if gate_type == "ID":
        return
    if gate_type == "X": qc.x(t[0]); return
    if gate_type == "Y": qc.y(t[0]); return
    if gate_type == "Z": qc.z(t[0]); return
    if gate_type == "H": qc.h(t[0]); return
    if gate_type == "S": qc.s(t[0]); return
    if gate_type == "T": qc.t(t[0]); return
    if gate_type.startswith("RX_PI_"):
        d = int(gate_type.split("_")[-1])
        qc.rx(np.pi/float(d), t[0]); return
    if gate_type.startswith("RY_PI_"):
        d = int(gate_type.split("_")[-1])
        qc.ry(np.pi/float(d), t[0]); return
    if gate_type.startswith("RZ_PI_"):
        d = int(gate_type.split("_")[-1])
        qc.rz(np.pi/float(d), t[0]); return
    if gate_type == "CX": qc.cx(t[0], t[1]); return
    if gate_type == "CZ": qc.cz(t[0], t[1]); return
    if gate_type == "SWAP": qc.swap(t[0], t[1]); return
    if gate_type == "CCX": qc.ccx(t[0], t[1], t[2]); return
    if gate_type == "CSWAP": qc.cswap(t[0], t[1], t[2]); return
    if gate_type == "CCZ": qc.h(t[2]); qc.ccx(t[0], t[1], t[2]); qc.h(t[2]); return


def build_gate_costs(vocab: List[str]) -> Dict[str, float]:
    """
    Creates a mapping {token: error_cost} for the entire vocabulary.
    """
    gate_costs: Dict[str, float] = {}
    
    for token in vocab:
        # 1. Parse the Gate Type from the token (e.g., "CCX_0_1_2" -> "CCX")
        gate_type = token.split("_")[0]
        
        # 2. Assign Cost based on Architecture Physics
        if gate_type == "ID":
            # Explicit ID usually implies 'Wait', which incurs decoherence (T1/T2).
            # However, for optimization, we often want to encourage ID (doing nothing)
            # over doing something. Let's keep it 0 or very low.
            cost = 0.0
            
        elif gate_type in ["X", "Y", "H"]:
            cost = COST_1Q_PHYSICAL
            
        elif gate_type in ["Z", "S", "S_dag", "T", "T_dag"]:
            cost = COST_VIRTUAL
            
        # 2Q Gates
        elif gate_type in ["CX", "CZ"]: 
            # Note: On some hardware CZ is native, on others CX is native.
            # Usually they are 1:1 convertible via Hadamards.
            cost = COST_2Q_NATIVE
            
        elif gate_type == "SWAP":
            cost = COST_SWAP
            
        # 3Q Gates (The heavy penalties)
        elif gate_type == "CCX":
            cost = COST_CCX
        elif gate_type == "CCZ":
            cost = COST_CCZ
        elif gate_type == "CSWAP":
            cost = COST_CSWAP
            
        # Rotations
        # RX and RY are physical pulses (X/Y axes)
        elif gate_type.startswith("RX") or gate_type.startswith("RY"):
            cost = COST_1Q_PHYSICAL
            
        # RZ is virtual
        elif gate_type.startswith("RZ"):
            cost = COST_VIRTUAL
            
        # Measure
        elif gate_type == "MEASURE":
            # Measurement is very noisy (readout error ~1-2%), 
            # but usually necessary. Penalizing it might discourage outputting results.
            # Best to set it low to avoid the model fearing measurement.
            cost = COST_1Q_PHYSICAL 
            
        # Special Tokens (<PAD>, <BOS>, <EOS>) -> 0 Cost
        else:
            cost = 0.0
            
        gate_costs[token] = cost
        
    return gate_costs
### Default Gate Costs (can be tuned per hardware)
# --- REALISTIC HARDWARE ERROR RATES (IBM Heron/Eagle) ---
# Loss = 1.0 means 100% failure probability.

# Virtual Gates (Frame Changes): Free and perfect.
COST_VIRTUAL = 0.0

# 1-Qubit Gates (SX, X, RZ): ~0.03% to 0.1% error
COST_1Q_PHYSICAL = 0.001  # 0.1% (Conservative estimate)

# 2-Qubit Gates (ECR/CZ/CNOT): ~0.8% to 1.5% error
# This is the dominant noise source.
COST_2Q_NATIVE = 0.01     # 1.0% error (Standard "Unit of Pain")

# Composite Gates (Decomposed into native gates)
COST_SWAP = 0.03          # SWAP ~= 3 CNOTs
COST_CCX = 0.06           # Toffoli ~= 6 CNOTs
COST_CCZ = 0.06
COST_CSWAP = 0.09