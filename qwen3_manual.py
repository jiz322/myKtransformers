#!/usr/bin/env python
# qwen3_manual.py
#
# Pure‑PyTorch inference for Qwen3‑30B‑A3B (Mixture‑of‑Experts)
# Requirements: torch>=2.1, safetensors>=0.5.3, tokenizers>=0.21
# Usage:
#   python qwen3_manual.py \
#       --model-dir ../LLMs/Qwen3-30B-A3B \
#       --prompt "Give me a short introduction to large language model." \
#       --device cuda:0 --dtype bfloat16 --max-new 200 --temperature 0.8
#
# ---------------------------------------------------------------

import argparse, json, math, os, time, gc
import torch
torch.set_grad_enabled(False)
from safetensors.torch import load_file
from tokenizers import Tokenizer

# -------------------------------- qwen3_manual.py (add at top) ---
TARGET_DTYPE = torch.bfloat16      # will be overwritten by CLI

def LIN(in_f, out_f, bias=False):        # 1 line helper
    return torch.nn.Linear(in_f, out_f, bias=bias,
                           device="cpu", dtype=TARGET_DTYPE)

def EMB(vocab, dim):
    return torch.nn.Embedding(vocab, dim, device="cpu", dtype=TARGET_DTYPE)
# -----------------------------------------------------------------



# ------------------ tiny utilities ------------------------------------------------
def get_rope(freqs, t, x):
    sin, cos = freqs
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    sin, cos = sin[t].unsqueeze(0), cos[t].unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# Rotary embedding (rope_theta comes from config) ----------------
class RotaryEmbedding:
    def __init__(self, dim, max_pos, theta=1e6, device="cpu", dtype=torch.float32):
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
        t = torch.arange(max_pos, device=device, dtype=dtype)
        freqs = torch.outer(t, inv_freq)          # [max_pos, dim/2]
        self.sin = freqs.sin()                    # buffers, not parameters
        self.cos = freqs.cos()

# Expert MLP ------------------------------------------------------
class ExpertMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gate_proj = LIN(in_dim, hidden_dim, bias=False)
        self.up_proj   = LIN(in_dim, hidden_dim, bias=False)
        self.down_proj = LIN(hidden_dim, in_dim, bias=False)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))

# Sparse‑MoE block ------------------------------------------------
class SparseMoe(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_experts=128, top_k=8):
        super().__init__()
        self.top_k = top_k
        self.gate = LIN(in_dim, num_experts, bias=False)
        self.experts = torch.nn.ModuleList([ExpertMLP(in_dim, hidden_dim) 
                                            for _ in range(num_experts)])

    def forward(self, x):
        # x: [B, T, D]
        gate_logits = self.gate(x)                        # [B, T, E]
        topk_scores, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)  # …,k
        gate_weights = torch.softmax(topk_scores, dim=-1)                    # probs
        out = torch.zeros_like(x)
        for i in range(self.top_k):
            idx = topk_idx[..., i]                       # [B, T]
            weight = gate_weights[..., i].unsqueeze(-1)  # [B,T,1]
            y = torch.stack([self.experts[e](tok) for e, tok in
                             zip(idx.view(-1), x.view(-1, x.size(-1)))])      # slow CPU
            y = y.view_as(x)
            out += weight * y
        return out

# Attention -------------------------------------------------------
class QwenAttention(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = cfg["hidden_size"] // self.n_heads
        self.q_proj = LIN(cfg["hidden_size"], cfg["hidden_size"]*2, bias=False)
        self.k_proj = LIN(cfg["hidden_size"], self.n_kv*self.head_dim, bias=False)
        self.v_proj = LIN(cfg["hidden_size"], self.n_kv*self.head_dim, bias=False)
        self.o_proj = LIN(cfg["hidden_size"]*2, cfg["hidden_size"], bias=False)
        self.q_norm = RMSNorm(self.head_dim*2, eps=cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])

    def forward(self, x, rope: RotaryEmbedding):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, 2*self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim)

        pos = torch.arange(T, device=x.device)
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = get_rope((rope.sin, rope.cos), pos, q)
        k = get_rope((rope.sin, rope.cos), pos, k)

        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        att = att.softmax(-1)
        out = att @ v                                 # [B,T,H,hd]
        out = out.view(B, T, -1)
        return self.o_proj(out)

# Decoder layer ---------------------------------------------------
class DecoderLayer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg["hidden_size"]
        self.self_attn = QwenAttention(cfg)
        self.mlp  = SparseMoe(h, h//2, cfg["num_experts"], cfg["num_experts_per_tok"])
        self.input_layernorm        = RMSNorm(h, eps=cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(h, eps=cfg["rms_norm_eps"])

    def forward(self, x, rope):
        h = x + self.self_attn(self.input_layernorm(x), rope)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


# --- small wrapper so every key starts with  “model.” -----------------
class QwenMoe(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = _QwenBody(cfg)           # everything except head
        self.lm_head = LIN(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def forward(self, input_ids):
        hidden = self.model(input_ids)
        return self.lm_head(hidden)
# Complete model --------------------------------------------------
class _QwenBody(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = EMB(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = torch.nn.ModuleList([DecoderLayer(cfg)
                                        for _ in range(cfg["num_hidden_layers"])])
        self.norm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.lm_head = LIN(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, self.rope)
        return self.norm(h)              # <-- end here

# ------------------ weight loader --------------------------------

def load_weights(model, model_dir, dtype=torch.bfloat16, device="cpu"):
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]

    # group tensor‑names per shard to avoid reopening the index repeatedly
    shard_to_names = {}
    for name, shard in weight_map.items():
        shard_to_names.setdefault(shard, []).append(name)

    param_dict = model.state_dict()

    for i, (shard, names) in enumerate(shard_to_names.items()):
        print(f"#{i} shard start loading")
        shard_path = os.path.join(model_dir, shard)
        print("start load_file")
        tensor_map = load_file(shard_path, device="cpu")     # 1⃣ load once on CPU
        for i, name in enumerate(names):
            tensor = tensor_map[name]
            if tensor.dtype != dtype:
                print(f"start the {i} to dtype")
                tensor = tensor.to(dtype)                    # 2⃣ convert in‑place
            param_dict[name].data.copy_(tensor)              # 3⃣ copy directly
            del tensor_map[name]                             # free asap
        del tensor_map                                       # free shard
        gc.collect()                                         # hint to Python
        torch.cuda.empty_cache()                             # in case device is CUDA

    return model

def group_by_shard(weight_map):
    by = {}
    for name, shard in weight_map.items():
        by.setdefault(shard, []).append(name)
    return by

# ------------------ generation loop ------------------------------
@torch.no_grad()
def generate(model, input_ids, max_new=128, temperature=0.8, top_p=0.95, top_k=50, eos=None):
    model.eval()
    for _ in range(max_new):
        logits = model(input_ids)[:, -1, :] / temperature   # [B,V]
        topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
        probs = torch.softmax(topk_vals, -1)
        next_token = topk_idx.gather(-1, torch.multinomial(probs, 1))
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if eos and next_token.item() == eos:
            break
    return input_ids

# ------------------ main -----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    print("Building model …")
    model = QwenMoe(cfg).to(args.device)
    print("Loading weights … (this takes a few minutes the first time)")
    load_weights(model, args.model_dir, dtype=dtype, device=device)

    input_ids = torch.tensor([tok.encode(args.prompt).ids], device=device)
    start = time.time()
    out = generate(model, input_ids, max_new=args.max_new, temperature=args.temperature,
                   eos=tok.token_to_id("<|im_end|>"))
    duration = time.time() - start

    print(f"==> {tok.decode(out[0].tolist())}")
    print(f"(generated {out.shape[1]-input_ids.shape[1]} tokens in {duration:.2f}s)")

if __name__ == "__main__":
    main()
