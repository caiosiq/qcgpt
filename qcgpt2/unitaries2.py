import torch
import math
from typing import List
from .gates2 import ID_TO_TOKEN2
from .circuits2 import Circuit2


def _I() -> torch.Tensor:
    return torch.eye(2, dtype=torch.complex64)

def _X() -> torch.Tensor:
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)

def _Y() -> torch.Tensor:
    return torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)

def _Z() -> torch.Tensor:
    return torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

def _H() -> torch.Tensor:
    return (1.0 / math.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64)

def _S() -> torch.Tensor:
    return torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64)

def _T() -> torch.Tensor:
    return torch.tensor([[1, 0], [0, torch.exp(1j * torch.tensor(math.pi / 4))]], dtype=torch.complex64)

def _RX(theta: float) -> torch.Tensor:
    c = math.cos(theta / 2)
    s = -1j * math.sin(theta / 2)
    return torch.tensor([[c, s], [s, c]], dtype=torch.complex64)

def _RY(theta: float) -> torch.Tensor:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.complex64)

def _RZ(theta: float) -> torch.Tensor:
    return torch.tensor([[torch.exp(-0.5j * torch.tensor(theta)), 0], [0, torch.exp(0.5j * torch.tensor(theta))]], dtype=torch.complex64)

def _two_qubit_unitary(name: str) -> torch.Tensor:
    # ... (Same as before: CX, CZ, SWAP) ...
    # Ensure SWAP is correct
    SWAP = torch.tensor([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=torch.complex64)
    if name == "CX":
        CX = torch.zeros((4, 4), dtype=torch.complex64)
        CX[0, 0] = 1; CX[1, 1] = 1; CX[2, 3] = 1; CX[3, 2] = 1
        return CX
    if name == "CZ":
        return torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64))
    if name == "SWAP":
        return SWAP
    raise ValueError(f"Unsupported two-qubit gate: {name}")

def _three_qubit_unitary(name: str) -> torch.Tensor:
    I8 = torch.eye(8, dtype=torch.complex64)
    if name == "CCX":
        # Toffoli: Control on q0, q1, Target q2
        U = I8.clone(); U[6, 7] = 1; U[7, 6] = 1; U[6, 6] = 0; U[7, 7] = 0; return U
    if name == "CCZ":
        U = I8.clone(); U[7, 7] = -1; return U
    if name == "CSWAP":
        # Fredkin: Control on q0, Swap q1, q2
        # |101> <-> |110>  (Index 5 <-> 6)
        U = I8.clone(); U[5, 6] = 1; U[6, 5] = 1; U[5, 5] = 0; U[6, 6] = 0; return U
    raise ValueError(f"Unsupported three-qubit gate: {name}")

def _lift_one_qubit(U1: torch.Tensor, target: int, n_qubits: int) -> torch.Tensor:
    D = 2 ** n_qubits
    U = torch.zeros((D, D), dtype=torch.complex64)
    for idx in range(D):
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
        x = bits[target]
        y_vec = U1[:, x]
        for y in range(2):
            out_bits = bits.copy()
            out_bits[target] = y
            out_index = 0
            for i in range(n_qubits):
                out_index |= (out_bits[i] << i)
            U[out_index, idx] += y_vec[y]
    return U

def _lift_two_qubit(U2: torch.Tensor, targets: List[int], n_qubits: int) -> torch.Tensor:
    a, b = targets
    if a == b:
        return torch.eye(2 ** n_qubits, dtype=torch.complex64)
    D = 2 ** n_qubits
    U = torch.zeros((D, D), dtype=torch.complex64)
    for idx in range(D):
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
        x = bits[a] * 2 + bits[b]
        y_vec = U2[:, x]
        for y_idx in range(4):
            out_bits = bits.copy()
            out_bits[a] = (y_idx >> 1) & 1
            out_bits[b] = y_idx & 1
            out_index = 0
            for i in range(n_qubits):
                out_index |= (out_bits[i] << i)
            U[out_index, idx] += y_vec[y_idx]
    return U

def _permute_three_qubit(U3: torch.Tensor, targets: List[int]) -> torch.Tensor:
    # ... (Same logic, critically relies on correct mapping) ...
    n_qubits = 3; D = 2 ** n_qubits
    P = torch.zeros((D, D), dtype=torch.complex64)
    # The base U3 is defined for qubits [0, 1, 2]
    # We want to map base qubit 0 -> targets[0], 1 -> targets[1], 2 -> targets[2]
    mapping = {0: targets[0], 1: targets[1], 2: targets[2]}
    for idx in range(D):
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
        out_bits = [0, 0, 0]
        for src in range(3):
            # mapping[src] is the physical qubit index where the base qubit 'src' lands
            out_bits[mapping[src]] = bits[src]
        out_index = 0
        for i in range(n_qubits):
            out_index |= (out_bits[i] << i)
        P[out_index, idx] = 1
    # U_physical = P @ U_base @ P.T
    return P @ U3 @ P.T

def _build_three_qubit_direct(name: str, targets: List[int], n_qubits: int) -> torch.Tensor:
    a, b, c = targets
    D = 2 ** n_qubits
    U = torch.zeros((D, D), dtype=torch.complex64)
    for idx in range(D):
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
        out_bits = bits.copy()
        phase = 1.0 + 0j
        if name == "CCX":
            if bits[a] == 1 and bits[b] == 1:
                out_bits[c] ^= 1
        elif name == "CCZ":
            if bits[a] == 1 and bits[b] == 1 and bits[c] == 1:
                phase = -1.0 + 0j
        elif name == "CSWAP":
            if bits[a] == 1:
                out_bits[b], out_bits[c] = out_bits[c], out_bits[b]
        out_index = 0
        for i in range(n_qubits):
            out_index |= (out_bits[i] << i)
        U[out_index, idx] += phase
    return U

def get_gate_unitary(gate_type: str, targets: List[int], n_qubits: int = 3) -> torch.Tensor:
    if gate_type in {"X", "Y", "Z", "H", "S", "T"}:
        base = {"X": _X(), "Y": _Y(), "Z": _Z(), "H": _H(), "S": _S(), "T": _T()}[gate_type]
        return _lift_one_qubit(base, targets[0], n_qubits)
    if gate_type.startswith("RX_"):
        angles = {"RX_PI_16": math.pi / 16, "RX_PI_8": math.pi / 8, "RX_PI_4": math.pi / 4, "RX_PI_2": math.pi / 2, "RX_PI": math.pi}
        return _lift_one_qubit(_RX(angles[gate_type]), targets[0], n_qubits)
    if gate_type.startswith("RY_"):
        angles = {"RY_PI_16": math.pi / 16, "RY_PI_8": math.pi / 8, "RY_PI_4": math.pi / 4, "RY_PI_2": math.pi / 2, "RY_PI": math.pi}
        return _lift_one_qubit(_RY(angles[gate_type]), targets[0], n_qubits)
    if gate_type.startswith("RZ_"):
        angles = {"RZ_PI_16": math.pi / 16, "RZ_PI_8": math.pi / 8, "RZ_PI_4": math.pi / 4, "RZ_PI_2": math.pi / 2, "RZ_PI": math.pi}
        return _lift_one_qubit(_RZ(angles[gate_type]), targets[0], n_qubits)
    if gate_type in {"CX", "CZ", "SWAP"}:
        return _lift_two_qubit(_two_qubit_unitary(gate_type), targets, n_qubits)
    if gate_type in {"CCX", "CCZ", "CSWAP"}:
        return _build_three_qubit_direct(gate_type, targets, n_qubits)
    if gate_type == "ID":
        return torch.eye(2 ** n_qubits, dtype=torch.complex64)
    raise ValueError(f"Unsupported gate type: {gate_type}")

def get_unitary_for_token_id(tok_id: int, n_qubits: int = 3) -> torch.Tensor:
    name = ID_TO_TOKEN2[tok_id]
    if name in {"<PAD>", "<BOS_CIRC>", "<EOS_CIRC>"}:
        return torch.eye(2 ** n_qubits, dtype=torch.complex64)
    parts = name.split("_")
    if parts[0] in {"RX", "RY", "RZ"} and len(parts) >= 4 and parts[1] == "PI":
        gate_type = f"{parts[0]}_PI_{parts[2]}"
        target = int(parts[3])
        return get_gate_unitary(gate_type, [target], n_qubits=n_qubits)
    for k in (3, 2, 1):
        if len(parts) > k:
            try:
                targets = list(map(int, parts[-k:]))
                gate_type = "_".join(parts[:-k])
                return get_gate_unitary(gate_type, targets, n_qubits=n_qubits)
            except ValueError:
                pass
    # Fallback: one-qubit gate without underscore target (shouldn't occur in vocab)
    return get_gate_unitary(parts[0], [], n_qubits=n_qubits)


def build_circuit_unitary2(circ: Circuit2, n_qubits: int = 3) -> torch.Tensor:
    U = torch.eye(2 ** n_qubits, dtype=torch.complex64)
    for g in circ.gates:
        if g.gate_type in {"<PAD>", "<BOS_CIRC>", "<EOS_CIRC>"}:
            continue
        G = get_gate_unitary(g.gate_type, g.targets, n_qubits)
        U = G @ U
    return U
