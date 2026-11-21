"""Compare Humming's packed token indices and padding with a scalar reference."""
import torch

from align_sm89 import align


def check(values, block_size):
    num_experts = 8
    sorted_ids, block_ids, total = align(torch.tensor(values, dtype=torch.int32), block_size, num_experts)
    expected_ids, expected_blocks = [], []
    for expert in range(num_experts):
        slots = [i for i, value in enumerate(values) if value == expert]
        padded = slots + [len(values)] * (-len(slots) % block_size)
        expected_ids.extend(padded)
        expected_blocks.extend([expert] * (len(padded) // block_size))
    assert total.tolist() == [len(expected_ids)]
    assert sorted_ids[:len(expected_ids)].tolist() == expected_ids
    assert block_ids[:len(expected_blocks)].tolist() == expected_blocks
    assert (block_ids[len(expected_blocks):] == -1).all()


if __name__ == "__main__":
    for block_size in (4, 16, 64):
        for values in ([], [-1] * 12, [3], [2, 1, 3, 1, 8, -1, 0, 2], list(range(8)) * 17):
            check(values, block_size)
