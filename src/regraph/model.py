"""ReGraph model assembly (`ReGraph.md` §2.2, §2.5).

    F_0 -> Γ_0 -> F_1 -> ... -> Γ_{T-1} -> F_T -> LMHead
    Γ_t(Z, H, P) = Replace(Z, I_B, Fuse(B, Read(B, H, P))),  B = Z[I_B]

The frozen decoder-only LLM is split into T+1 consecutive layer groups; a
Read-Fuse-Replace round runs between adjacent groups, touching only the graph-query
token positions. The graph is never serialized into the prompt.

The manual layer loop is written against transformers==4.57.3 `modeling_llama.py`
(pinned in requirements.txt): `LlamaDecoderLayer.forward` returns a plain tensor,
takes `past_key_values`/`position_embeddings`, and gradient checkpointing is applied
inside `GradientCheckpointingLayer.__call__` when enabled on the model.

Precision (00-conventions.md): frozen Llama runs in its native dtype (bf16) outside
autocast; trainable modules keep fp32 parameters and run under
`torch.autocast(bfloat16)`; node softmax and diffusion run in fp32 inside the reader.

Masking: one additive-causal-mask builder is used for training, prefill, and decode,
so the three paths are numerically consistent (and batched right-padded generation is
position-correct). Fully padded query rows attend to their own slot to avoid NaNs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from regraph.modules.fuse import Fuse
from regraph.modules.graph_encoder import build_graph_encoder
from regraph.modules.reader import TopologyDiffusedReader
from regraph.modules.roles import RoleEmbedding


@dataclass
class ReGraphOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


def compute_group_bounds(num_layers: int, num_rounds: int) -> list[tuple[int, int]]:
    """T+1 consecutive groups over `num_layers`; remainder goes to earlier groups
    (docs/components/03-query-tokens.md §3.2). num_rounds == T."""
    num_groups = num_rounds + 1
    assert 1 <= num_groups <= num_layers
    base, rem = divmod(num_layers, num_groups)
    bounds, lo = [], 0
    for g in range(num_groups):
        hi = lo + base + (1 if g < rem else 0)
        bounds.append((lo, hi))
        lo = hi
    assert bounds[-1][1] == num_layers
    return bounds


class ReGraph(nn.Module):
    """See docs/components/06-model.md. Trainable: graph encoder, b_base, reader,
    fuse, role embedding. Frozen: every original LLM parameter."""

    def __init__(self, cfg: dict, llm, tokenizer, placeholder_id: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.llm = llm
        self.tokenizer = tokenizer

        for p in self.llm.parameters():  # CLAUDE.md rule 3: freeze, never no_grad
            p.requires_grad_(False)
        if cfg["llm"].get("gradient_checkpointing", False):
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        d_llm = self.llm.config.hidden_size
        assert d_llm == cfg["llm"]["d_llm"], (
            f"config d_llm={cfg['llm']['d_llm']} but model hidden_size={d_llm}"
        )
        self.d_llm = d_llm
        self.num_query_tokens = cfg["model"]["num_query_tokens"]
        self.num_rounds = cfg["model"]["num_rounds"]
        self.group_bounds = compute_group_bounds(
            self.llm.config.num_hidden_layers, self.num_rounds
        )

        if placeholder_id is None:
            placeholder_id = tokenizer.convert_tokens_to_ids(
                cfg["model"]["placeholder_token"]
            )
        assert placeholder_id is not None and placeholder_id >= 0
        self.placeholder_id = placeholder_id

        # --- trainable modules (fp32 parameters) --------------------------------
        self.graph_encoder = build_graph_encoder(cfg)
        self.role_emb = RoleEmbedding(cfg["graph_encoder"]["d_graph"])

        # B_base initialized from the empirical distribution of the frozen embedding
        # matrix so the tokens start in-distribution (03-query-tokens.md §3.1)
        embed_w = self.llm.get_input_embeddings().weight
        b_base = torch.empty(self.num_query_tokens, d_llm, dtype=torch.float32)
        b_base.normal_(mean=embed_w.float().mean().item(), std=embed_w.float().std().item())
        self.b_base = nn.Parameter(b_base)

        reader_kwargs = dict(
            d_llm=d_llm,
            d_graph=cfg["graph_encoder"]["d_graph"],
            d_reader=cfg["reader"]["d_reader"],
            heads=cfg["reader"]["heads"],
            max_hops=cfg["reader"]["max_hops"],
            shared_hop_weights=cfg["reader"]["shared_hop_weights"],
            w_o_init=cfg["reader"]["w_o_init"],
            w_o_init_std=cfg["reader"]["w_o_init_std"],
        )
        self.share_reader = cfg["reader"]["share_across_rounds"]
        if self.share_reader:
            self.reader = TopologyDiffusedReader(**reader_kwargs)
        else:
            self.reader = nn.ModuleList(
                TopologyDiffusedReader(**reader_kwargs) for _ in range(self.num_rounds)
            )
        # NOTE: `ReGraph.md` §2.4 applies exactly one dropout to the evidence, inside
        # Fuse: B_post = B_pre + Diag(g) Dropout(R). An additional dropout on R here
        # would double-drop the graph signal (0.1 twice = 0.19 effective), which is not
        # what the spec states. `reader.dropout` therefore has no output-side effect;
        # Fuse owns the single Dropout(R). See docs/OPEN-QUESTIONS.md Q18.

        self.share_fuse = cfg["fuse"]["share_across_rounds"]
        if self.share_fuse:
            self.fuse = Fuse(d_llm=d_llm, dropout=cfg["fuse"]["dropout"])
        else:
            self.fuse = nn.ModuleList(
                Fuse(d_llm=d_llm, dropout=cfg["fuse"]["dropout"])
                for _ in range(self.num_rounds)
            )

    # ------------------------------------------------------------------ helpers

    @property
    def device(self) -> torch.device:
        return self.b_base.device

    @property
    def llm_dtype(self) -> torch.dtype:
        return self.llm.get_input_embeddings().weight.dtype

    def _autocast(self):
        """Trainable modules: fp32 params under bf16 autocast when the LLM is bf16;
        plain fp32 otherwise (tiny fp32 test models)."""
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.llm_dtype == torch.bfloat16,
        )

    def _reader_at(self, t: int) -> TopologyDiffusedReader:
        return self.reader if self.share_reader else self.reader[t]

    def _fuse_at(self, t: int) -> Fuse:
        return self.fuse if self.share_fuse else self.fuse[t]

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ----------------------------------------------------------- input assembly

    def build_inputs_embeds(
        self, input_ids: torch.Tensor, b_positions: torch.Tensor
    ) -> torch.Tensor:
        """Embed real tokens with the frozen embed_tokens, then scatter b_base into
        the b_positions slots (docs/components/01-data.md, 03-query-tokens.md §3.4)."""
        at_slots = input_ids.gather(1, b_positions)
        assert bool((at_slots == self.placeholder_id).all()), (
            "b_positions must point at the placeholder token id"
        )
        embeds = self.llm.get_input_embeddings()(input_ids)
        bsz = input_ids.shape[0]
        b_idx = torch.arange(bsz, device=input_ids.device).unsqueeze(1)
        b_base = self.b_base.to(embeds.dtype).unsqueeze(0).expand(bsz, -1, -1)
        return embeds.index_put((b_idx, b_positions), b_base)

    def _additive_causal_mask(
        self,
        attention_mask: torch.Tensor,  # [B, past_len + q_len], 1 = real
        past_len: int,
        q_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """[B, 1, q_len, kv_len] additive mask: 0 = attend, finfo.min = blocked.
        Causal over absolute positions; padding columns blocked; fully blocked
        (padded) query rows keep their own slot to avoid NaN activations."""
        device = attention_mask.device
        kv_len = attention_mask.shape[1]
        assert kv_len == past_len + q_len
        j = torch.arange(kv_len, device=device)
        i = torch.arange(q_len, device=device).unsqueeze(1) + past_len
        causal = j.unsqueeze(0) <= i                                   # [q, kv]
        allowed = causal.unsqueeze(0) & attention_mask.bool().unsqueeze(1)
        allowed = allowed | (j.unsqueeze(0) == i).unsqueeze(0)         # self-slot
        mask = torch.zeros(
            attention_mask.shape[0], 1, q_len, kv_len, dtype=dtype, device=device
        )
        return mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)

    # ------------------------------------------------------------- graph memory

    def encode_graph(self, batch: dict) -> dict:
        """Node memory H and transition edges, computed once per forward pass and
        reused by all T rounds (docs/components/02-graph-encoder.md)."""
        node_mask = batch["node_mask"]
        bsz, n_max = node_mask.shape
        with self._autocast():
            h_flat = self.graph_encoder(
                batch["x"], batch["edge_index"], batch["edge_attr"]
            )
            h_base, _ = to_dense_batch(
                h_flat, batch["node_batch"], batch_size=bsz, max_num_nodes=n_max
            )
            # h_v = h_v^base + e_role(r_v(q)); zeroed at ~node_mask (§2.1)
            h = h_base + self.role_emb(batch["roles"])
            h = h * node_mask.unsqueeze(-1)

            ctx = {
                "node_mask": node_mask,
                "edge_src_pad": batch["edge_src_pad"],
                "edge_dst_pad": batch["edge_dst_pad"],
                "edge_w": batch["edge_w"].float(),
                "h": h,
            }
            if self.share_reader:
                # K_H, V_H do not depend on t — computed once (04-reader.md §4.4)
                ctx["k_h"], ctx["v_h"] = self.reader.precompute(h)
        return ctx

    def _graph_round(
        self,
        t: int,
        hidden: torch.Tensor,
        b_positions: torch.Tensor,
        ctx: dict,
        diagnostics: dict | None,
        return_reading: bool = False,
    ) -> torch.Tensor:
        """Γ_t: Replace(Z, I_B, Fuse(B, Read(B, H, P))) (`ReGraph.md` §2.3-2.4)."""
        bsz = hidden.shape[0]
        b_idx = torch.arange(bsz, device=hidden.device).unsqueeze(1)
        b_pre = hidden[b_idx, b_positions]                       # gather [B, N_B, d]
        assert b_pre.dtype == hidden.dtype

        with self._autocast():
            reader = self._reader_at(t)
            if self.share_reader:
                k_h, v_h = ctx["k_h"], ctx["v_h"]
            else:
                k_h, v_h = reader.precompute(ctx["h"])
            r, read_diag = reader(
                b_pre,
                k_h,
                v_h,
                ctx["node_mask"],
                ctx["edge_src_pad"],
                ctx["edge_dst_pad"],
                ctx["edge_w"],
                return_reading=return_reading,
            )
            b_post, gate = self._fuse_at(t)(b_pre, r)   # §2.4 applies the single Dropout(R)

        b_post = b_post.to(hidden.dtype)
        if diagnostics is not None:
            diagnostics.setdefault("gates", []).append(gate.detach())
            diagnostics.setdefault("alphas", []).append(read_diag["alpha"])
            diagnostics.setdefault("s0_entropy", []).append(read_diag["s0_entropy"])
            diagnostics.setdefault("s_tilde_entropy", []).append(
                read_diag["s_tilde_entropy"]
            )
            if return_reading:
                diagnostics.setdefault("s_tilde", []).append(read_diag["s_tilde"])
        # Replace: out-of-place index_put keeps autograd clean under checkpointing
        return hidden.index_put((b_idx, b_positions), b_post)

    # ------------------------------------------------------------- layer looping

    def _run_layers(
        self,
        hidden: torch.Tensor,
        causal_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: DynamicCache | None,
        use_cache: bool,
        graph: tuple[torch.Tensor, dict, dict | None, bool] | None,
    ) -> torch.Tensor:
        """F_0 -> Γ_0 -> ... -> Γ_{T-1} -> F_T -> final norm. `graph=None` runs the
        plain LLM stack (decode steps run no Γ_t at all — 08-inference.md §8.3)."""
        position_embeddings = self.llm.model.rotary_emb(hidden, position_ids)
        for t, (lo, hi) in enumerate(self.group_bounds):
            for layer in self.llm.model.layers[lo:hi]:
                hidden = layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            if graph is not None and t < self.num_rounds:
                b_positions, ctx, diagnostics, return_reading = graph
                hidden = self._graph_round(
                    t, hidden, b_positions, ctx, diagnostics, return_reading
                )
        return self.llm.model.norm(hidden)

    # ------------------------------------------------------------------ training

    def forward(self, batch: dict, return_reading: bool = False) -> ReGraphOutput:
        batch = self._to_device(batch)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        b_positions = batch["b_positions"]
        bsz, seq_len = input_ids.shape

        ctx = self.encode_graph(batch) if self.num_rounds > 0 else None
        embeds = self.build_inputs_embeds(input_ids, b_positions)

        causal_mask = self._additive_causal_mask(
            attention_mask, past_len=0, q_len=seq_len, dtype=embeds.dtype
        )
        cache_position = torch.arange(seq_len, device=embeds.device)
        # right padding: HF's own scheme (positions 0..S-1) is correct for real tokens
        position_ids = cache_position.unsqueeze(0)

        diagnostics: dict = {}
        graph = (
            (b_positions, ctx, diagnostics, return_reading)
            if self.num_rounds > 0
            else None
        )
        hidden = self._run_layers(
            embeds, causal_mask, position_ids, cache_position,
            past_key_values=None, use_cache=False, graph=graph,
        )
        logits = self.llm.lm_head(hidden)

        loss = None
        if "labels" in batch:
            # §3.2 answer-only next-token likelihood; -100 masks everything else
            shift_logits = logits[:, :-1].float()
            shift_labels = batch["labels"][:, 1:]
            if (shift_labels != -100).any():
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
            else:  # fully masked batch: loss is exactly 0 (07-training.md test 2)
                loss = shift_logits.sum() * 0.0
        return ReGraphOutput(loss=loss, logits=logits, diagnostics=diagnostics)

    # ----------------------------------------------------------------- inference

    @torch.no_grad()
    def generate(
        self,
        batch: dict,
        max_new_tokens: int = 32,
        collect_diagnostics: bool = False,
        return_reading: bool = False,
    ) -> dict:
        """Prefill runs all Γ_t once; decoding is a standard KV-cached greedy loop
        (`ReGraph.md` §2.5 "Efficient inference with KV caching", 08-inference.md).
        Right-padded batches: position ids and cache offsets come from attention_mask.
        """
        self.eval()
        batch = self._to_device(batch)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].clone()
        b_positions = batch["b_positions"]
        bsz, prompt_len = input_ids.shape
        device = input_ids.device

        ctx = self.encode_graph(batch) if self.num_rounds > 0 else None
        embeds = self.build_inputs_embeds(input_ids, b_positions)

        past = DynamicCache(config=self.llm.config)
        causal_mask = self._additive_causal_mask(
            attention_mask, past_len=0, q_len=prompt_len, dtype=embeds.dtype
        )
        cache_position = torch.arange(prompt_len, device=device)
        position_ids = cache_position.unsqueeze(0)
        diagnostics: dict | None = {} if (collect_diagnostics or return_reading) else None
        graph = (
            (b_positions, ctx, diagnostics, return_reading)
            if self.num_rounds > 0
            else None
        )
        hidden = self._run_layers(
            embeds, causal_mask, position_ids, cache_position,
            past_key_values=past, use_cache=True, graph=graph,
        )
        # logits of y_1 sit at each row's last *real* position
        n_real = attention_mask.sum(-1)                                  # [B]
        last = (n_real - 1).view(bsz, 1, 1).expand(-1, 1, hidden.shape[-1])
        next_logits = self.llm.lm_head(hidden.gather(1, last).squeeze(1))

        eos_ids = self._eos_ids()
        generated = torch.empty(bsz, 0, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        for step in range(max_new_tokens):
            next_token = next_logits.argmax(-1)                          # greedy
            next_token = torch.where(
                finished, torch.full_like(next_token, eos_ids[0]), next_token
            )
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= torch.isin(next_token, eos_ids)
            if bool(finished.all()) or step == max_new_tokens - 1:
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones(bsz, 1, dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
            token_embeds = self.llm.get_input_embeddings()(next_token).unsqueeze(1)
            kv_len = attention_mask.shape[1]
            causal_mask = self._additive_causal_mask(
                attention_mask, past_len=kv_len - 1, q_len=1, dtype=token_embeds.dtype
            )
            cache_position = torch.tensor([kv_len - 1], device=device)
            position_ids = (n_real + step).unsqueeze(1)                  # per row
            # decode: no Γ_t — the cached B-token states already carry the evidence
            hidden = self._run_layers(
                token_embeds, causal_mask, position_ids, cache_position,
                past_key_values=past, use_cache=True, graph=None,
            )
            next_logits = self.llm.lm_head(hidden[:, -1])

        out = {"ids": generated, "texts": self._decode(generated, eos_ids)}
        if diagnostics is not None:
            out["diagnostics"] = diagnostics
        return out

    @torch.no_grad()
    def generate_naive(self, batch: dict, max_new_tokens: int = 32) -> dict:
        """Reference path: re-runs the entire interleaved forward on the full
        sequence at every step (08-inference.md acceptance test 1). Same position
        scheme as `generate`, so outputs must be token-identical."""
        self.eval()
        batch = self._to_device(batch)
        input_ids = batch["input_ids"]
        prompt_mask = batch["attention_mask"]
        b_positions = batch["b_positions"]
        bsz, prompt_len = input_ids.shape
        device = input_ids.device

        ctx = self.encode_graph(batch) if self.num_rounds > 0 else None
        prompt_embeds = self.build_inputs_embeds(input_ids, b_positions)
        n_real = prompt_mask.sum(-1)

        eos_ids = self._eos_ids()
        generated = torch.empty(bsz, 0, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        for step in range(max_new_tokens):
            gen_embeds = (
                self.llm.get_input_embeddings()(generated)
                if generated.shape[1]
                else prompt_embeds[:, :0]
            )
            embeds = torch.cat([prompt_embeds, gen_embeds], dim=1)
            seq_len = embeds.shape[1]
            attention_mask = torch.cat(
                [prompt_mask, torch.ones(bsz, step, dtype=prompt_mask.dtype, device=device)],
                dim=1,
            )
            causal_mask = self._additive_causal_mask(
                attention_mask, past_len=0, q_len=seq_len, dtype=embeds.dtype
            )
            cache_position = torch.arange(seq_len, device=device)
            prompt_pos = cache_position[:prompt_len].unsqueeze(0).expand(bsz, -1)
            gen_pos = n_real.unsqueeze(1) + torch.arange(step, device=device)
            position_ids = torch.cat([prompt_pos, gen_pos], dim=1)

            graph = (b_positions, ctx, None, False) if self.num_rounds > 0 else None
            hidden = self._run_layers(
                embeds, causal_mask, position_ids, cache_position,
                past_key_values=None, use_cache=False, graph=graph,
            )
            if step == 0:
                last = (n_real - 1).view(bsz, 1, 1).expand(-1, 1, hidden.shape[-1])
                next_logits = self.llm.lm_head(hidden.gather(1, last).squeeze(1))
            else:
                next_logits = self.llm.lm_head(hidden[:, -1])

            next_token = next_logits.argmax(-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, eos_ids[0]), next_token
            )
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= torch.isin(next_token, eos_ids)
            if bool(finished.all()):
                break

        return {"ids": generated, "texts": self._decode(generated, eos_ids)}

    # ------------------------------------------------------------------- private

    def _eos_ids(self) -> torch.Tensor:
        ids = {self.tokenizer.eos_token_id}
        gen_eos = getattr(self.llm.generation_config, "eos_token_id", None)
        if gen_eos is not None:
            ids.update(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
        return torch.tensor(sorted(i for i in ids if i is not None), device=self.device)

    def _decode(self, generated: torch.Tensor, eos_ids: torch.Tensor) -> list[str]:
        texts = []
        eos_set = set(eos_ids.tolist())
        for row in generated.tolist():
            keep = []
            for tok in row:
                if tok in eos_set:
                    break
                keep.append(tok)
            texts.append(self.tokenizer.decode(keep, skip_special_tokens=True).strip())
        return texts

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if torch.is_tensor(v) else v
        return out


def load_regraph(cfg: dict, device: str = "cuda") -> ReGraph:
    """Build ReGraph around the frozen bf16 backbone named in the config."""
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg["llm"]["dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm"]["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = cfg["model"]["pad_token"]
        assert tokenizer.pad_token_id is not None
    llm = AutoModelForCausalLM.from_pretrained(cfg["llm"]["name"], dtype=dtype)
    llm.to(device)
    assert llm.config.num_hidden_layers == cfg["llm"]["num_layers"]
    model = ReGraph(cfg, llm, tokenizer).to(device)
    # trainable modules stay fp32 on purpose (00-conventions.md): .to(device) above
    # moves them; the frozen LLM was already cast at load time.
    return model
