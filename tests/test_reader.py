"""04-reader.md acceptance tests for the Topology-Diffused Graph Reader."""

import math

import torch

from conftest import D_GRAPH, D_LLM, dense_p, make_batch
from regraph.modules.reader import TopologyDiffusedReader
from regraph.modules.transition import build_transition_edges


def make_reader(max_hops=2, heads=4, shared_hop=False, seed=0) -> TopologyDiffusedReader:
    torch.manual_seed(seed)
    return TopologyDiffusedReader(
        d_llm=D_LLM, d_graph=D_GRAPH, d_reader=D_GRAPH, heads=heads,
        max_hops=max_hops, shared_hop_weights=shared_hop,
    ).eval()


def randomize(reader: TopologyDiffusedReader, seed=1) -> TopologyDiffusedReader:
    """Un-zero W_O / W_alpha so the full path is exercised."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        reader.w_o.weight.copy_(torch.randn(reader.w_o.weight.shape, generator=g) * 0.1)
        reader.w_alpha.weight.copy_(
            torch.randn(reader.w_alpha.weight.shape, generator=g) * 0.5
        )
        reader.w_alpha.bias.copy_(torch.randn(reader.w_alpha.bias.shape, generator=g))
    return reader


def run_reader(reader, batch, b_pre, return_reading=True):
    h = torch.randn(batch["node_mask"].shape[0], batch["node_mask"].shape[1], D_GRAPH)
    h = h * batch["node_mask"].unsqueeze(-1)
    k_h, v_h = reader.precompute(h)
    r, diag = reader(
        b_pre, k_h, v_h, batch["node_mask"],
        batch["edge_src_pad"], batch["edge_dst_pad"], batch["edge_w"],
        return_reading=return_reading,
    )
    return r, diag, h


def test_s_tilde_row_stochastic_masked_batch():
    torch.manual_seed(0)
    batch = make_batch()
    reader = randomize(make_reader())
    b_pre = torch.randn(3, 4, D_LLM)
    _, diag, _ = run_reader(reader, batch, b_pre)
    s = diag["s_tilde"]
    assert torch.allclose(s.sum(-1), torch.ones_like(s.sum(-1)), atol=1e-5)
    # padded nodes carry exactly zero mass
    assert (s[~batch["node_mask"][:, None, None, :].expand_as(s > 0)] == 0).all()


def test_k0_reduces_to_masked_cross_attention():
    torch.manual_seed(0)
    batch = make_batch()
    reader = randomize(make_reader(max_hops=0))
    b_pre = torch.randn(3, 4, D_LLM)
    r, diag, h = run_reader(reader, batch, b_pre)

    # reference: plain masked multi-head cross-attention with the same weights
    heads, hd = reader.heads, reader.head_dim
    bsz, n_max = batch["node_mask"].shape
    u = reader.w_q(reader.ln_b(b_pre)).view(bsz, 4, heads, hd).permute(0, 2, 1, 3)
    k = reader.w_k(reader.ln_h(h)).view(bsz, n_max, heads, hd).permute(0, 2, 1, 3)
    v = reader.w_v(reader.ln_h(h)).view(bsz, n_max, heads, hd).permute(0, 2, 1, 3)
    logits = u @ k.transpose(-1, -2) / math.sqrt(hd)
    logits = logits.masked_fill(~batch["node_mask"][:, None, None, :], torch.finfo(logits.dtype).min)
    attn = logits.softmax(-1)
    ref = reader.w_o((attn @ v).permute(0, 2, 1, 3).reshape(bsz, 4, heads * hd))
    assert torch.allclose(r, ref, atol=1e-5)


def test_p_identity_makes_all_hops_equal():
    """Self-loop-only edge list: s_k == s_0 for all k, so s_tilde == s_0."""
    torch.manual_seed(0)
    batch = make_batch(n_edges=(0, 0, 0))  # only self-loops in P
    reader = randomize(make_reader(max_hops=2))
    b_pre = torch.randn(3, 4, D_LLM)
    _, diag, h = run_reader(reader, batch, b_pre)
    with torch.no_grad():  # rebuild s0 with the same q/k path
        k_h, _ = reader.precompute(h)
        u = reader._split_heads(reader.w_q(reader.ln_b(b_pre)))
        logits = u @ k_h.transpose(-1, -2) / math.sqrt(reader.head_dim)
        logits = logits.masked_fill(
            ~batch["node_mask"][:, None, None, :], torch.finfo(logits.dtype).min
        )
        s0 = logits.float().softmax(-1)
    assert torch.allclose(diag["s_tilde"], s0, atol=1e-6)


def test_dense_reference_full_reader():
    """s_tilde == sum_k alpha_k * (s0 @ P^k) built densely (fp32, atol 1e-5)."""
    torch.manual_seed(3)
    n = 10
    batch = make_batch(n_qs=(4,), n_nodes=(n,), n_edges=(18,), seed=5)
    reader = randomize(make_reader(max_hops=2), seed=7)
    b_pre = torch.randn(1, 4, D_LLM)
    _, diag, h = run_reader(reader, batch, b_pre)

    src, dst, w = build_transition_edges(
        batch["edge_index"], n, symmetrize=True, add_self_loops=True
    )
    p = dense_p(src, dst, w, n)
    with torch.no_grad():
        k_h, _ = reader.precompute(h)
        u = reader._split_heads(reader.w_q(reader.ln_b(b_pre)))
        logits = u @ k_h.transpose(-1, -2) / math.sqrt(reader.head_dim)
        logits = logits.masked_fill(
            ~batch["node_mask"][:, None, None, :], torch.finfo(logits.dtype).min
        )
        s0 = logits.float().softmax(-1)
        alpha = reader.w_alpha(reader.ln_b(b_pre)).float()
        alpha = alpha.view(1, 4, reader.heads, 3).softmax(-1).permute(0, 2, 1, 3)
        ref = torch.zeros_like(s0)
        for k in range(3):
            ref = ref + alpha[..., k].unsqueeze(-1) * (s0 @ torch.matrix_power(p, k))
    assert torch.allclose(diag["s_tilde"], ref, atol=1e-5)


def test_masking_invariant_to_padding_growth():
    """Results unchanged when n_max grows (extra padded nodes)."""
    torch.manual_seed(0)
    b1 = make_batch(n_qs=(4,), n_nodes=(6,), n_edges=(9,), seed=11)
    b2 = make_batch(n_qs=(4, 4), n_nodes=(6, 16), n_edges=(9, 20), seed=11)
    reader = randomize(make_reader())
    b_pre = torch.randn(1, 4, D_LLM)

    h1 = torch.randn(1, 6, D_GRAPH)
    n_max2 = b2["node_mask"].shape[1]
    h2 = torch.zeros(2, n_max2, D_GRAPH)
    h2[0, :6] = h1[0]

    k1, v1 = reader.precompute(h1)
    r1, d1 = reader(
        b_pre, k1, v1, b1["node_mask"], b1["edge_src_pad"], b1["edge_dst_pad"],
        b1["edge_w"], return_reading=True,
    )
    k2, v2 = reader.precompute(h2)
    r2, d2 = reader(
        b_pre.expand(2, -1, -1).contiguous(), k2, v2, b2["node_mask"],
        b2["edge_src_pad"], b2["edge_dst_pad"], b2["edge_w"], return_reading=True,
    )
    assert torch.allclose(r1[0], r2[0], atol=1e-5)
    assert torch.allclose(d1["s_tilde"][0], d2["s_tilde"][0, :, :, :6].reshape_as(d1["s_tilde"][0])
                          if n_max2 == 6 else d2["s_tilde"][0][..., :6], atol=1e-5)
    assert (d2["s_tilde"][0][..., 6:] == 0).all()


def test_zero_init_evidence_and_first_step():
    """r_evidence is exactly zero at step 0; after one optimizer step it is not."""
    torch.manual_seed(0)
    batch = make_batch()
    reader = make_reader()  # zero-init W_O
    reader.train()
    b_pre = torch.randn(3, 4, D_LLM)
    r, _, h = run_reader(reader, batch, b_pre, return_reading=False)
    assert (r == 0).all()
    opt = torch.optim.SGD(reader.parameters(), lr=1.0)
    k_h, v_h = reader.precompute(h)
    r, _ = reader(
        b_pre, k_h, v_h, batch["node_mask"],
        batch["edge_src_pad"], batch["edge_dst_pad"], batch["edge_w"],
    )
    r.sum().backward()
    assert reader.w_o.weight.grad is not None and reader.w_o.weight.grad.abs().sum() > 0
    opt.step()
    r2, _ = reader(
        b_pre, k_h.detach(), v_h.detach(), batch["node_mask"],
        batch["edge_src_pad"], batch["edge_dst_pad"], batch["edge_w"],
    )
    assert r2.abs().sum() > 0


def test_uniform_hops_at_init():
    batch = make_batch()
    reader = make_reader()
    b_pre = torch.randn(3, 4, D_LLM)
    _, diag, _ = run_reader(reader, batch, b_pre, return_reading=False)
    assert torch.allclose(diag["alpha"], torch.full_like(diag["alpha"], 1 / 3), atol=1e-6)


def test_zero_init_w_o_starves_attention_gradient():
    """Documents the pathology behind docs/OPEN-QUESTIONS.md Q17.

    R = S~ V_H W_O, so dL/dS~ = (dL/dR)(V_H W_O)^T. With W_O zero-initialized the
    node-selection weights W_Q/W_K receive *exactly zero* gradient at step 0."""
    torch.manual_seed(0)
    batch = make_batch()
    reader = make_reader()  # w_o_init defaults to zeros
    b_pre = torch.randn(3, 4, D_LLM)
    h = torch.randn(3, batch["node_mask"].shape[1], D_GRAPH) * batch["node_mask"].unsqueeze(-1)
    k_h, v_h = reader.precompute(h)
    r, _ = reader(b_pre, k_h, v_h, batch["node_mask"],
                  batch["edge_src_pad"], batch["edge_dst_pad"], batch["edge_w"])
    r.sum().backward()
    assert reader.w_o.weight.grad.abs().sum() > 0          # W_O itself learns
    assert reader.w_q.weight.grad.abs().sum() == 0         # but attention does not
    assert reader.w_k.weight.grad.abs().sum() == 0


def test_normal_init_w_o_opens_attention_gradient():
    """`reader.w_o_init: normal` restores gradient flow to W_Q/W_K at step 0."""
    torch.manual_seed(0)
    batch = make_batch()
    reader = TopologyDiffusedReader(
        d_llm=D_LLM, d_graph=D_GRAPH, d_reader=D_GRAPH, heads=4, max_hops=2,
        w_o_init="normal", w_o_init_std=1e-3,
    )
    b_pre = torch.randn(3, 4, D_LLM)
    h = torch.randn(3, batch["node_mask"].shape[1], D_GRAPH) * batch["node_mask"].unsqueeze(-1)
    k_h, v_h = reader.precompute(h)
    r, _ = reader(b_pre, k_h, v_h, batch["node_mask"],
                  batch["edge_src_pad"], batch["edge_dst_pad"], batch["edge_w"])
    r.sum().backward()
    for name, w in (("w_q", reader.w_q), ("w_k", reader.w_k), ("w_o", reader.w_o)):
        assert w.weight.grad.abs().sum() > 0, f"{name} still starved"
    # the perturbation is still small at init, preserving near-frozen-LLM behavior
    assert r.abs().mean() < 0.5


def test_invalid_w_o_init_rejected():
    import pytest

    with pytest.raises(ValueError, match="w_o_init"):
        TopologyDiffusedReader(d_llm=D_LLM, d_graph=D_GRAPH, d_reader=D_GRAPH,
                               heads=4, w_o_init="xavier")
