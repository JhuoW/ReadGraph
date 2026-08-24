"""05-fuse.md acceptance tests for Gated Residual Evidence Fusion."""

import torch

from regraph.modules.fuse import Fuse

D = 32


def make_fuse(dropout=0.0) -> Fuse:
    torch.manual_seed(0)
    return Fuse(d_llm=D, dropout=dropout).eval()


def test_shapes_and_gate_range():
    fuse = make_fuse()
    with torch.no_grad():
        fuse.w_g.weight.normal_()
        fuse.w_g.bias.normal_()
    b = torch.randn(2, 4, D)
    r = torch.randn(2, 4, D)
    b_post, gate = fuse(b, r)
    assert b_post.shape == b.shape
    assert gate.shape == (2, 4, 1)
    assert (gate > 0).all() and (gate < 1).all()


def test_zero_evidence_is_identity():
    fuse = make_fuse()
    with torch.no_grad():
        fuse.w_g.weight.normal_()
    b = torch.randn(2, 4, D)
    b_post, _ = fuse(b, torch.zeros_like(b))
    assert torch.equal(b_post, b)


def test_identity_at_init_with_zero_reader_output():
    """Reader W_O zero-init -> r = 0 -> Fuse exactly identity, gate ≈ 0.5."""
    fuse = make_fuse()
    b = torch.randn(2, 4, D)
    b_post, gate = fuse(b, torch.zeros_like(b))
    assert torch.equal(b_post, b)
    assert torch.allclose(gate, torch.full_like(gate, 0.5))


def test_gate_extremes():
    b = torch.randn(2, 4, D)
    r = torch.randn(2, 4, D)
    fuse = make_fuse()
    with torch.no_grad():
        fuse.w_g.bias.fill_(30.0)
    b_post, gate = fuse(b, r)
    assert torch.allclose(b_post, b + r, atol=1e-5)
    with torch.no_grad():
        fuse.w_g.bias.fill_(-30.0)
    b_post, gate = fuse(b, r)
    assert torch.allclose(b_post, b, atol=1e-5)


def test_per_token_independence():
    fuse = make_fuse()
    with torch.no_grad():
        fuse.w_g.weight.normal_()
    b = torch.randn(1, 6, D)
    r = torch.randn(1, 6, D)
    base, _ = fuse(b, r)
    r2 = r.clone()
    r2[0, 3] += 1.0
    out, _ = fuse(b, r2)
    changed = (out != base).any(-1)[0]
    assert changed[3] and changed.sum() == 1


def test_injects_raw_r_not_normalized():
    """The injected term equals gate * r, NOT gate * ln_r(r)."""
    fuse = make_fuse()
    with torch.no_grad():
        fuse.w_g.weight.normal_(std=0.3)
        fuse.w_g.bias.fill_(0.7)
        # make ln_r visibly different from identity
        fuse.ln_r.weight.fill_(3.0)
        fuse.ln_r.bias.fill_(1.0)
    b = torch.randn(2, 4, D)
    r = torch.randn(2, 4, D) * 5 + 2  # un-normalized scale/shift
    b_post, gate = fuse(b, r)
    # hand-computed: gate from normalized concat, injection raw
    ref_gate = torch.sigmoid(
        fuse.w_g(torch.cat([fuse.ln_b(b), fuse.ln_r(r)], dim=-1).float())
    )
    assert torch.allclose(gate, ref_gate)
    assert torch.allclose(b_post, b + gate * r, atol=1e-6)
    assert not torch.allclose(b_post, b + gate * fuse.ln_r(r), atol=1e-2)


def test_gate_gradient_flows_at_init():
    """Zero-init w_g must still receive gradient (05-fuse.md §5.3)."""
    fuse = Fuse(d_llm=D, dropout=0.0)
    b = torch.randn(2, 4, D, requires_grad=True)
    r = torch.randn(2, 4, D)
    b_post, gate = fuse(b, r)
    b_post.sum().backward()
    assert fuse.w_g.weight.grad is not None
    assert fuse.w_g.weight.grad.abs().sum() > 0
