"""Compare the SM89 full-prefill kernel with the original across the bias boundary."""
import os

import torch
from vllm.models.inkling2.nvidia.ops.attention_sm90 import MODE_OUT_LSE, rel_attention


def check(tokens, context, window=0):
    heads, kv_heads, dim, page, extent = 24, 8, 128, 128, 1024
    blocks = (context + page - 1) // page
    cache = torch.randn(blocks, kv_heads, page, 2 * dim, device="cuda", dtype=torch.bfloat16)
    keys, values = cache.to(torch.float8_e4m3fn).view(torch.uint8).transpose(1, 2).split(dim, dim=-1)
    query = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(tokens + 128, heads, extent + 256, device="cuda", dtype=torch.bfloat16)
    pages = torch.randperm(blocks, device="cuda", dtype=torch.int32)[None, :]
    lengths = torch.tensor([context], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)

    def evaluate(enabled):
        os.environ["INKLING_BALANCED_DSPARK"] = enabled
        out = torch.empty_like(query, dtype=torch.float32)
        lse = torch.empty(heads, tokens, device="cuda", dtype=torch.float32)
        rel_attention(
            query, keys, values, out, lse, bias, block_table=pages, cache_seqlens=lengths,
            cu_seqlens_q=offsets, max_query_len=tokens, softmax_scale=dim**-0.5,
            window=window, num_splits=1, mode=MODE_OUT_LSE,
        )
        return out, lse

    original, original_lse = evaluate("0")
    balanced, balanced_lse = evaluate("1")
    if window:
        assert torch.equal(original, balanced) and torch.equal(original_lse, balanced_lse)
    else:
        torch.testing.assert_close(balanced, original, atol=0.002, rtol=0.004)
        torch.testing.assert_close(balanced_lse, original_lse, atol=2e-5, rtol=1e-5)


if __name__ == "__main__":
    assert torch.cuda.get_device_capability() == (8, 9)
    torch.manual_seed(217)
    for context in (1023, 1024, 1025, 1215, 1216, 1217, 8192):
        check(129, context)
    check(640, 32768)
    check(129, 4096, window=1024)
    print("SM89 prefill outputs and log-sum-exp matched the original within tolerance")
