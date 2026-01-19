import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qcgpt3 import GateRegistry
from qcgpt3.data.dataset import HighPerformanceDataset

def validate_dataset():
    print("\n--- Test 6: Dataset Pipeline & Augmentations ---")
    
    registry = GateRegistry(n_qubits=3)
    
    # Initialize Dataset with Augmentations
    dataset = HighPerformanceDataset(
        registry=registry,
        qdpe=None, # CPU mode
        num_samples=10,
        n_qubits=3,
        raw_max_depth=10,
        augment_commutation=True,
        augment_permutation=True
    )
    
    print(f"Dataset initialized. Length: {len(dataset)}")
    
    # 1. Fetch Sample
    print("Fetching sample 0...")
    sample = dataset[0]
    
    spec = sample["spec_tensor"]
    tokens = sample["circ_tokens"]
    ref_circ = sample["ref_circuit"]
    
    print(f"Spec Shape: {spec.shape}")     # Should be (8, 2, 8, 2)
    print(f"Tokens Shape: {tokens.shape}") # (L,)
    print(f"Circuit Gates: {len(ref_circ.gates)}")
    
    if spec.shape != (8, 2, 8, 2):
        print(f"FAIL: Spec tensor shape mismatch. Got {spec.shape}")
        sys.exit(1)
        
    # 2. Verify Augmentation (Randomness)
    # Fetch same index again. HighPerformanceDataset uses deterministic seed based on index?
    # dataset[idx] sets seed = (idx * magic) % ...
    # So dataset[0] should always be the same circuit BEFORE augmentation if logic is deterministic.
    # BUT wait, the code says:
    # rng = np.random.RandomState(seed)
    # ... apply_commutation_jitter(..., rng)
    # So it should be deterministic per index.
    
    # To test augmentation variety, we should check if DIFFERENT indices produce valid data.
    # Or we can check if disabling determinism (if possible) or just checking that we get variety across indices.
    
    print("Checking variety across samples...")
    tokens_list = []
    for i in range(5):
        s = dataset[i]
        t = s["circ_tokens"]
        tokens_list.append(t)
        
    # Check if they are identical (unlikely for random circuits)
    all_same = all(torch.equal(tokens_list[0], x) for x in tokens_list[1:])
    if all_same:
        print("WARN: All 5 samples were identical. Random generation might be broken or seed collision.")
    else:
        print("PASS: Samples show variety.")

    # 3. Visualization of Spec Tensor
    # Spec Tensor contains input/output state pairs.
    # Shape: (N_pairs, 2 (In/Out), Dim, 2 (Re/Im))
    # Let's visualize the Output states (index 1) for the first few pairs.
    # Since inputs are basis states |0>, |1>, ..., the outputs are columns of U.
    # So we are visualizing the columns of the Unitary.
    
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    spec_np = spec.numpy()
    
    # Construct the Unitary from the spec
    # U_re = spec[:, 1, :, 0]
    # U_im = spec[:, 1, :, 1]
    # But wait, spec is (N_pairs, 2, Dim, 2)
    # Pair i corresponds to Input |i> (row i of Identity)
    # The Output is U|i>, which is the i-th COLUMN of U.
    # So spec[i, 1, :, :] is the i-th column.
    # If we stack them, we get U^T (since we stack rows).
    
    U_re = spec_np[:, 1, :, 0]
    U_im = spec_np[:, 1, :, 1]
    U_rec = U_re + 1j * U_im
    # U_rec is (Rows=Pairs, Cols=BasisDim).
    # Since Pair i is Column i, U_rec is U^T.
    U_final = U_rec.T
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(U_final.real, cmap='coolwarm')
    plt.title("Re(U) from Spec")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(U_final.imag, cmap='coolwarm')
    plt.title("Im(U) from Spec")
    plt.colorbar()
    
    save_path = os.path.join(output_dir, "dataset_spec_vis.png")
    plt.savefig(save_path)
    print(f"Spec visualization saved to: {save_path}")
    print("PASS: Dataset pipeline functional.")

if __name__ == "__main__":
    validate_dataset()
