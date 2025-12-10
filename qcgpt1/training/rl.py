import torch
from typing import Tuple

from .rollouts import (
    build_batch_specs,
    compute_reward_for_circuit,
    compute_reward_qiskit_blackbox,
    RewardBaseline,
)
from ..models.policy import CircuitPolicy
from ..encoding import tokens_to_circuit
from ..gates import BOS_CIRC_ID, EOS_CIRC_ID, PAD_ID


def rl_step(
    model: CircuitPolicy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    max_len: int,
    lambda_len: float,
    max_gates_ref: int,
    baseline: RewardBaseline,
    use_qiskit_fidelity: bool = False,
    use_blackbox: bool = False,
    method: str = "statevector",
    use_noise: bool = False,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[float, float]:
    """
    Performs one REINFORCE update step.

    Returns:
      mean_reward, loss_value
    """
    model.train()

    spec_states_batch, spec_batch, spec_pad_mask = build_batch_specs(
        batch_size=batch_size,
        max_gates_ref=max_gates_ref,
    )
    spec_batch = spec_batch.to(device)
    spec_pad_mask = spec_pad_mask.to(device)

    # 2) Sample circuits from policy (keep graph for log-prob gradients)
    sampled_tokens, log_probs = model.sample_circuit_tokens(
        spec_batch,
        spec_pad_mask,
        bos_id=BOS_CIRC_ID,
        eos_id=EOS_CIRC_ID,
        max_len=max_len,
    )
        # sampled_tokens: [B, L]
        # log_probs:      [B]

    # 3) Convert tokens to Circuits and compute rewards
    B, L = sampled_tokens.shape
    rewards_list = []
    for i in range(B):
        seq = sampled_tokens[i].tolist()
        # strip PAD tokens; BOS/EOS handled inside tokens_to_circuit
        seq = [t for t in seq if t != PAD_ID]
        circ = tokens_to_circuit(seq)

        if use_blackbox:
            R = compute_reward_qiskit_blackbox(
                spec_states_batch[i],
                circ,
                lambda_len=lambda_len,
                method=method,
                use_noise=use_noise,
                p1=p1,
                p2=p2,
            )
        else:
            R = compute_reward_for_circuit(
                spec_states_batch[i],
                circ,
                lambda_len=lambda_len,
            )
        rewards_list.append(R)

    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
    mean_reward = rewards.mean().item()

    # 4) REINFORCE loss: -(R - baseline) * log_pi
    advantages = rewards - baseline.value
    loss = -(advantages * log_probs).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    baseline.update(mean_reward)

    return mean_reward, loss.item()


def train_rl(
    model: CircuitPolicy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int = 10_000,
    batch_size: int = 16,
    max_len: int = 32,
    lambda_len: float = 0.1,
    max_gates_ref: int = 6,
    log_every: int = 100,
    use_qiskit_fidelity: bool = False,
    use_blackbox: bool = False,
    method: str = "statevector",
    use_noise: bool = False,
    p1: float = 0.0,
    p2: float = 0.0,
):
    baseline = RewardBaseline(momentum=0.9)

    for step in range(1, num_steps + 1):
        mean_R, loss_val = rl_step(
            model=model,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            max_len=max_len,
            lambda_len=lambda_len,
            max_gates_ref=max_gates_ref,
            baseline=baseline,
            use_qiskit_fidelity=use_qiskit_fidelity,
            use_blackbox=use_blackbox,
            method=method,
            use_noise=use_noise,
            p1=p1,
            p2=p2,
        )

        if step % log_every == 0:
            print(
                f"[RL] Step {step:05d}  "
                f"MeanReward={mean_R:.4f}  "
                f"Baseline={baseline.value:.4f}  "
                f"Loss={loss_val:.4f}  "
                f"Blackbox={use_blackbox}  Noise={use_noise}  Method={method}"
            )