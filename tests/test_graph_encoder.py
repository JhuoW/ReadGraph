"""02-graph-encoder.md acceptance tests: node memory, roles, transition operator."""

import torch
from torch_geometric.utils import to_dense_batch

from conftest import D_ATTR, D_GRAPH, dense_p, make_batch, make_items, make_model, tiny_cfg
from regraph.modules.graph_encoder import build_graph_encoder
from regraph.modules.transition import build_transition_edges, diffuse_once


def test_h_shape_and_padding_zeroed():
    model = make_model()
    batch = make_batch()
    ctx = model.encode_graph(batch)
    bsz, n_max = batch["node_mask"].shape
    assert ctx["h"].shape == (bsz, n_max, D_GRAPH)
    assert (ctx["h"][~batch["node_mask"]] == 0).all()


def test_row_stochastic_identity_diffusion():
    """diffuse_once on S = I gives P; every row of P sums to 1."""
    torch.manual_seed(0)
    n = 9
    edge_index = torch.randint(0, n, (2, 15))
    src, dst, w = build_transition_edges(edge_index, n)
    s = torch.eye(n)
    out = diffuse_once(s.T.contiguous(), src, dst, w).T  # rows of S are columns of s_flat
    assert torch.allclose(out.sum(-1), torch.ones(n), atol=1e-6)


def test_diffuse_matches_dense_reference():
    torch.manual_seed(1)
    n, r = 12, 5
    edge_index = torch.randint(0, n, (2, 20))
    src, dst, w = build_transition_edges(edge_index, n)
    p = dense_p(src, dst, w, n)
    assert torch.allclose(p.sum(-1), torch.ones(n), atol=1e-6)
    s = torch.rand(r, n)
    s = s / s.sum(-1, keepdim=True)
    ref = s @ p
    got = diffuse_once(s.T.contiguous(), src, dst, w).T
    assert torch.allclose(got, ref, atol=1e-6)


def test_one_hot_diffuses_to_p_row():
    torch.manual_seed(2)
    n = 8
    edge_index = torch.randint(0, n, (2, 12))
    src, dst, w = build_transition_edges(edge_index, n)
    p = dense_p(src, dst, w, n)
    for u in (0, 3, 7):
        e_u = torch.zeros(1, n)
        e_u[0, u] = 1.0
        got = diffuse_once(e_u.T.contiguous(), src, dst, w).T
        assert torch.allclose(got[0], p[u], atol=1e-6)


def test_self_loops_guarantee_valid_p():
    src, dst, w = build_transition_edges(torch.zeros(2, 0, dtype=torch.long), 4)
    p = dense_p(src, dst, w, 4)
    assert torch.allclose(p, torch.eye(4))


def test_symmetrize_flag():
    edge_index = torch.tensor([[0], [1]])
    src_s, dst_s, _ = build_transition_edges(edge_index, 2, symmetrize=True)
    pairs = set(zip(src_s.tolist(), dst_s.tolist()))
    assert (1, 0) in pairs and (0, 1) in pairs
    src_d, dst_d, _ = build_transition_edges(edge_index, 2, symmetrize=False)
    pairs_d = set(zip(src_d.tolist(), dst_d.tolist()))
    assert (1, 0) not in pairs_d


def test_encoder_padding_independence():
    """Batch of 3 graphs gives the same h per graph as encoding each alone."""
    cfg = tiny_cfg()
    torch.manual_seed(0)
    enc = build_graph_encoder(cfg).eval()
    items = make_items(n_nodes=(6, 3, 10), n_edges=(9, 2, 14))
    batch = make_batch()
    with torch.no_grad():
        h_flat = enc(batch["x"], batch["edge_index"], batch["edge_attr"])
        h_batch, _ = to_dense_batch(
            h_flat, batch["node_batch"], batch_size=3,
            max_num_nodes=int(batch["num_nodes"].max()),
        )
        for i, it in enumerate(items):
            h_alone = enc(it["x"], it["edge_index"], it["edge_attr"])
            assert torch.allclose(h_batch[i, : h_alone.shape[0]], h_alone, atol=1e-4)


def test_role_embedding_zero_at_init():
    model = make_model()
    batch = make_batch()
    assert (model.role_emb.emb.weight == 0).all()
    with torch.no_grad():
        ctx = model.encode_graph(batch)
        h_flat = model.graph_encoder(batch["x"], batch["edge_index"], batch["edge_attr"])
        h_base, _ = to_dense_batch(
            h_flat, batch["node_batch"], batch_size=3,
            max_num_nodes=batch["node_mask"].shape[1],
        )
        h_base = h_base * batch["node_mask"].unsqueeze(-1)
    assert torch.allclose(ctx["h"], h_base, atol=1e-6)


def test_empty_edge_graph_encodes():
    cfg = tiny_cfg()
    enc = build_graph_encoder(cfg)
    x = torch.randn(5, D_ATTR)
    h = enc(x, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, D_ATTR))
    assert h.shape == (5, D_GRAPH) and torch.isfinite(h).all()
