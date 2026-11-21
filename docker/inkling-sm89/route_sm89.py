"""Pack the small routing tables used by single-request SM89 decode and verification."""
import torch
import triton
import triton.language as tl


@triton.jit
def _max(a, b):
    return tl.maximum(a, b)


@triton.jit
def _pack_small(
    Ids, Sorted, Owners, Count, N: tl.constexpr, WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_TILE: tl.constexpr, EXPERTS: tl.constexpr,
):
    block = tl.program_id(0)
    lanes = tl.arange(0, WIDTH)
    raw = tl.load(Ids + lanes, mask=lanes < N, other=-1)
    # Sort by expert, then by original slot. Invalid slots sort last.
    keys = tl.sort(tl.where((raw >= 0) & (raw < EXPERTS), raw, EXPERTS) * WIDTH + lanes)
    expert = keys // WIDTH
    prev = tl.gather(expert, tl.maximum(0, lanes - 1), 0)
    start = (lanes == 0) | (expert != prev)
    run = tl.associative_scan(tl.where(start, lanes, 0), 0, _max)
    # Each program emits one padded expert block. Repeated experts may span blocks.
    boundary = (expert != EXPERTS) & ((lanes - run) % BLOCK_SIZE == 0)
    rank = tl.cumsum(boundary.to(tl.int32), 0) - 1
    begin = tl.max(tl.where(boundary & (rank == block), lanes, -1), 0)
    owner = tl.sum(tl.where(lanes == begin, expert + 1, 0), 0) - 1
    offsets = tl.arange(0, BLOCK_TILE)
    index = begin + offsets
    key = tl.gather(keys, tl.minimum(tl.maximum(index, 0), WIDTH - 1), 0)
    valid = (begin >= 0) & (index < WIDTH) & (key // WIDTH == owner)
    slots = tl.where(valid, key % WIDTH, N)
    tl.store(Sorted + block * BLOCK_SIZE + offsets, slots, mask=offsets < BLOCK_SIZE)
    tl.store(Owners + block, owner)
    if block == 0:
        tl.store(Count, tl.sum(boundary.to(tl.int32), 0) * BLOCK_SIZE)


def align_small(ids, block_size, num_experts):
    n = ids.numel()
    max_slots = n + num_experts * (block_size - 1)
    capacity = min(n, triton.cdiv(max_slots, block_size))
    sorted_ids = ids.new_empty(capacity * block_size, dtype=torch.int32)
    owners = ids.new_empty(capacity, dtype=torch.int32)
    count = ids.new_empty(1, dtype=torch.int32)
    _pack_small[(capacity,)](
        ids, sorted_ids, owners, count, n, triton.next_power_of_2(n),
        block_size, triton.next_power_of_2(block_size), num_experts, num_warps=1,
    )
    return sorted_ids, owners, count
