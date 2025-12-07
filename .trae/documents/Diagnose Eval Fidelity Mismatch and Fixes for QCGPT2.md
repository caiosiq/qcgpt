## Likely Causes

* Incorrect fidelity formula: using elementwise contraction instead of Tr(U†V).

* PAD/BOS/EOS filtering bug in eval: filtering by `t != 0` instead of `PAD_ID2`, and not removing BOS/EOS.

* Distribution mismatch: training data angles vs eval angles; need to confirm that `pi/16, pi/8, pi/4` are included consistently.

* Spec encoding mismatch: ensure eval uses the pair-wise spec builder to match the encoder.

* Decoding behavior: max length/EOS handling may append PADs or early-stop inconsistently.

## Fixes to Implement

* Update fidelity to `trace = torch.trace(U_ref.conj().T @ U_cand)` and `fid = |trace|^2 / d^2`.

* In `eval_policy2.py` and `eval_grid2.py`, filter tokens using `PAD_ID2`, and drop BOS/EOS explicitly before building circuits.

* Ensure eval uses `qcgpt2.data.qiskit_utils2.sample_task2` and `qcgpt2.data.specs2.build_spec_sequence_batch` (pair-wise format) — audit both files.

* Add a quick self-check: identical circuits yield fidelity ≈ 1; differing by known rotation yield expected lower fidelity.

* Add logging in training to report token angle distribution seen per epoch to confirm fine-angle coverage.

* Verify decode strategy stops at EOS and respects `max_len`.

## Validation

* Run a small eval (e.g., 20 examples) and confirm average fidelity increases when CE loss is low.

* Unit tests: compare fidelity for identical circuits and for a permutation case; assert correct values.

* Spot-check that PAD/BOS/EOS filtering removes special tokens and only gate tokens remain.

## Deliverables

* Code fixes in eval scripts (fidelity formula, token filtering, imports).

* Optional training logs for token distribution.

* Short test script to sanity-check fidelity calculation.

