"""Build Humming expert blocks on SM89 for prompt chunks and single-request decoding."""
import torch


def align(ids, block_size, num_experts):
    if ids.is_cuda and ids.is_contiguous() and 0 < ids.numel() <= 64:
        from .route_sm89 import align_small

        return align_small(ids, block_size, num_experts)
    sorted_experts, order = ids.flatten().sort(stable=True)
    num_slots = order.numel()
    experts = torch.arange(num_experts + 1, device=ids.device, dtype=ids.dtype)
    starts = torch.searchsorted(sorted_experts, experts)
    counts = starts.diff()
    blocks = (counts + block_size - 1) // block_size
    ends = blocks.cumsum(0)

    # Size outputs from the input shape to avoid a device-to-host count on each layer.
    max_slots = num_slots + num_experts * (block_size - 1)
    max_blocks = min(num_slots, (max_slots + block_size - 1) // block_size)
    block = torch.arange(max_blocks, device=ids.device)
    owner = torch.searchsorted(ends, block, right=True)
    expert = owner.clamp_max(num_experts - 1)
    first_block = (ends - blocks)[expert]
    slot = (block - first_block)[:, None] * block_size + torch.arange(block_size, device=ids.device)

    keep = (owner < num_experts)[:, None] & (slot < counts[expert, None])
    index = (starts[expert, None] + slot).clamp(0, num_slots - 1)
    sorted_ids = torch.where(keep, order[index], num_slots).flatten()
    block_ids = owner.masked_fill(owner == num_experts, -1)
    return sorted_ids.int(), block_ids.int(), (ends[-1:] * block_size).int()
