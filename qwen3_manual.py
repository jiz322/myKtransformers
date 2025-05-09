#!/usr/bin/env python
# qwen3_manual.py  ―  Pure‑PyTorch (+IPEX) inference for Qwen3‑30B‑A3B
#
#   • Works on Sapphire‑Rapids/W‑re‑badged Xeon with AMX (BF16 / INT8 tiles)
#   • No CUDA dependency – everything runs on the CPU
#   • Handles the dtype‑mismatch bug that caused the original RuntimeError
#   • One‑liner to switch dtype/precision/temperature from the CLI
#
# Requirements:
#   pip install torch==2.7.0+cpu intel_extension_for_pytorch==2.7.0 safetensors tokenizers
#
# Example:
#   DNNL_MAX_CPU_ISA=AMX ONEDNN_VERBOSE=1 \
#   python qwen3_manual.py \
#       --model-dir ~/LLMs/Qwen3-30B-A3B \
#       --prompt "Give me a short introduction to large language model." \
#       --dtype bfloat16 --max-new 64 --temperature 0.8
# ---------------------------------------------------------------------------

import argparse, json, math, os, time, gc
import torch
import intel_extension_for_pytorch as ipex   # noqa: F401  (needed to enable AMX kernels)
from safetensors.torch import load_file
from tokenizers import Tokenizer

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')   # allow BF16 matmuls when possible

# ---------------------------------------------------------------------------
TARGET_DTYPE = torch.bfloat16                # overwritten by CLI

def LIN(in_f, out_f, bias=False):
    return torch.nn.Linear(in_f, out_f, bias=bias,
                           device='cpu', dtype=TARGET_DTYPE)

def EMB(vocab, dim):
    return torch.nn.Embedding(vocab, dim, device='cpu', dtype=TARGET_DTYPE)

# ‑‑‑‑‑ tiny helpers ‑‑‑‑‑----------------------------------------------------

# ---------------------------------------------------------------------------
# Correct RoPE broadcast:  sin,cos -> [1, T, 1, Hd/2]
def get_rope(freqs, positions, x):
    """
    x:  [B, T, H, Hd]   (even Hd)
    freqs: (sin, cos) where each is [max_pos, Hd/2]
    positions: LongTensor [T]  (0,1,2,...T‑1)
    """
    sin, cos = freqs
    sin = sin[positions].unsqueeze(0).unsqueeze(2).to(x.dtype)   # [1,T,1,Hd/2]
    cos = cos[positions].unsqueeze(0).unsqueeze(2).to(x.dtype)

    x1, x2 = x[..., 0::2], x[..., 1::2]                          # split last dim
    return torch.cat([x1 * cos - x2 * sin,                      # rotate pairs
                      x1 * sin + x2 * cos], dim=-1)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=TARGET_DTYPE))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# Rotary embedding ----------------------------------------------------------
class RotaryEmbedding:
    def __init__(self, dim, max_pos, theta=1e6):
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)          # [max_pos, dim/2]
        self.sin = freqs.sin().to(TARGET_DTYPE)
        self.cos = freqs.cos().to(TARGET_DTYPE)

# Expert MLP ----------------------------------------------------------------
class ExpertMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gate_proj = LIN(in_dim, hidden_dim, bias=False)
        self.up_proj   = LIN(in_dim, hidden_dim, bias=False)
        self.down_proj = LIN(hidden_dim, in_dim, bias=False)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))

# Sparse‑MoE block ----------------------------------------------------------
class SparseMoe(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_experts=128, top_k=8):
        super().__init__()
        self.top_k = top_k
        self.gate = LIN(in_dim, num_experts, bias=False)
        self.experts = torch.nn.ModuleList([ExpertMLP(in_dim, hidden_dim)
                                            for _ in range(num_experts)])

    def forward(self, x):
        gate_logits = self.gate(x)                        # [B, T, E]
        topk_scores, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        gate_weights = torch.softmax(topk_scores, dim=-1).to(x.dtype)
        out = torch.zeros_like(x)
        # token‑parallel expert execution (simple but OK for demo)
        for i in range(self.top_k):
            idx = topk_idx[..., i]
            weight = gate_weights[..., i].unsqueeze(-1)
            flat_x = x.view(-1, x.size(-1))
            expert_out = torch.stack([self.experts[e](tok)
                                      for e, tok in zip(idx.view(-1), flat_x)], dim=0)
            out += weight * expert_out.view_as(x)   # (B,T,1) * (B,T,D)  ->  broadcast OK

        return out

# Attention -----------------------------------------------------------------
class QwenAttention(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg['num_attention_heads']
        self.n_kv    = cfg['num_key_value_heads']
        self.head_dim = cfg['head_dim']            # 128

        self.q_proj = LIN(cfg['hidden_size'], self.n_heads * self.head_dim, bias=False)
        self.k_proj = LIN(cfg['hidden_size'], self.n_kv   * self.head_dim, bias=False)
        self.v_proj = LIN(cfg['hidden_size'], self.n_kv   * self.head_dim, bias=False)
        self.o_proj = LIN(self.n_heads * self.head_dim, cfg['hidden_size'], bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=cfg['rms_norm_eps'])
        self.k_norm = RMSNorm(self.head_dim, eps=cfg['rms_norm_eps'])

    # ---------------------------------------------------------------------
    # QwenAttention.forward (drop-in replacement)
    def forward(self, x, rope: RotaryEmbedding):
        """
        x  : [B, T, D]
        out: [B, T, D]
        """
        B, T, _ = x.shape
        Hd = self.head_dim
        H  = self.n_heads          # 32
        K  = self.n_kv             # 4

        # 1) Projections --------------------------------------------------
        q = self.q_proj(x).view(B, T, H, Hd)          # [B,T,H,Hd]
        k = self.k_proj(x).view(B, T, K, Hd)          # [B,T,K,Hd]
        v = self.v_proj(x).view(B, T, K, Hd)          # [B,T,K,Hd]

        # 2) RoPE ---------------------------------------------------------
        pos = torch.arange(T, device=x.device)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = get_rope((rope.sin, rope.cos), pos, q)
        k = get_rope((rope.sin, rope.cos), pos, k)

        # 3) Head layout: (B,H,T,Hd) vs (B,H,T,Hd)
        q = q.permute(0, 2, 1, 3)                     # [B,H,T,Hd]
        # broadcast K/V heads to all Q heads
        k = k.permute(0, 2, 1, 3)                     # [B,K,T,Hd]
        v = v.permute(0, 2, 1, 3)                     # [B,K,T,Hd]
        if K == 1 or K == H:
            k = k
            v = v
        else:
            # repeat K/V to match 32 heads (32 // 4 == 8 times)
            repeat = H // K
            k = k.repeat_interleave(repeat, dim=1)    # [B,H,T,Hd]
            v = v.repeat_interleave(repeat, dim=1)

        # 4) Scaled dot-product ------------------------------------------
        att = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(Hd)  # [B,H,T,T]

        # causal mask: keep lower-triangular, set upper to -inf
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(causal_mask, float('-inf'))

        att = att.softmax(-1)                         # still FP32
        out = (att @ v.float()).to(q.dtype)           # [B,H,T,Hd]  -> BF16

        # 5) Merge heads --------------------------------------------------
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, H*Hd)  # [B,T,D]
        return self.o_proj(out)


# Decoder layer --------------------------------------------------------------
class DecoderLayer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg['hidden_size']
        self.self_attn = QwenAttention(cfg)
        self.mlp  = SparseMoe(
            in_dim=h,
            hidden_dim=cfg['moe_intermediate_size'],
            num_experts=cfg['num_experts'],
            top_k=cfg['num_experts_per_tok'],
        )
        self.input_layernorm          = RMSNorm(h, eps=cfg['rms_norm_eps'])
        self.post_attention_layernorm = RMSNorm(h, eps=cfg['rms_norm_eps'])

    def forward(self, x, rope):
        h = x + self.self_attn(self.input_layernorm(x), rope)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h

# Model wrappers -------------------------------------------------------------
class _QwenBody(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = EMB(cfg['vocab_size'], cfg['hidden_size'])
        self.rope = RotaryEmbedding(
            dim=cfg['head_dim'],
            max_pos=cfg['max_position_embeddings'],
            theta=cfg['rope_theta'],
        )
        self.layers = torch.nn.ModuleList([DecoderLayer(cfg)
                                           for _ in range(cfg['num_hidden_layers'])])
        self.norm = RMSNorm(cfg['hidden_size'], eps=cfg['rms_norm_eps'])

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids).to(TARGET_DTYPE)
        for layer in self.layers:
            h = layer(h, self.rope)
        return self.norm(h)

class QwenMoe(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = _QwenBody(cfg)
        self.lm_head = LIN(cfg['hidden_size'], cfg['vocab_size'], bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))

# ---------------- weight loader --------------------------------------------
def load_weights(model, model_dir):
    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    weight_map = json.load(open(idx_path))['weight_map']
    shard_to_names = {}
    for name, shard in weight_map.items():
        shard_to_names.setdefault(shard, []).append(name)

    param_dict = model.state_dict()

    for i, (shard_file, names) in enumerate(shard_to_names.items()):
        print(f'#{i}  loading {shard_file} …')
        tensors = load_file(os.path.join(model_dir, shard_file), device='cpu')
        for name in names:
            param = param_dict[name]
            t = tensors[name]
            if t.dtype != TARGET_DTYPE:
                t = t.to(TARGET_DTYPE)
            if param.shape != t.shape:
                raise RuntimeError(f'Shape mismatch for {name}')
            param.data.copy_(t)
            del tensors[name]
        del tensors
        gc.collect()
    return model

# ---------------- token sampling -------------------------------------------
@torch.no_grad()
def generate(model, input_ids, max_new=16, temperature=0.7,
             top_p=0.95, top_k=50, eos=None):
    for _ in range(max_new):
        logits = model(input_ids)[:, -1, :]
        if temperature == 0:                       # greedy
            next_token = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            top_k = min(top_k, logits.size(-1))
            topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
            probs = torch.softmax(topk_vals, -1)
            next_token = topk_idx.gather(-1, torch.multinomial(probs, 1))
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if eos is not None and next_token.item() == eos:
            break
    return input_ids

# ---------------- main ------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--prompt',      default='Hello')
    ap.add_argument('--dtype',       default='bfloat16', choices=['bfloat16', 'float32'])
    ap.add_argument('--max-new',     type=int, default=64)
    ap.add_argument('--temperature', type=float, default=0.8)
    args = ap.parse_args()

    global TARGET_DTYPE
    TARGET_DTYPE = getattr(torch, args.dtype)

    cfg  = json.load(open(os.path.join(args.model_dir, 'config.json')))
    tok  = Tokenizer.from_file(os.path.join(args.model_dir, 'tokenizer.json'))

    print('Building model …')
    model = QwenMoe(cfg).to(memory_format=torch.channels_last)  # friendlier to AMX
    print('Loading weights … (first time takes a few minutes)')
    load_weights(model, args.model_dir)

    input_ids = torch.tensor([tok.encode(args.prompt).ids], dtype=torch.long)
    start = time.time()
    out_ids = generate(model, input_ids,
                       max_new=args.max_new,
                       temperature=args.temperature,
                       eos=tok.token_to_id('<|im_end|>'))
    t = time.time() - start

    text = tok.decode(out_ids[0].tolist())
    print('\n' + '='*80)
    print(text)
    print(f'--- generated {out_ids.size(1)-input_ids.size(1)} tokens in {t:.2f}s ---')

if __name__ == '__main__':
    main()
