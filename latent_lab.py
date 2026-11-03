#!/usr/bin/env python3
"""latent_lab.py -- a hands-on lab for latent communication between LLM agents.

Companion to the guide "How AI Agents Can Talk Without Words: Latent
Communication with KV Caches and Thought Vectors" and to The Architect's AI
Workshop episode of the same name.

The original guide (§10) references a script of this name that was not
published with it. This is an independent rebuild of the same experiments,
written from the guide's text: same five ideas, same toy scale, numbers
computed fresh by this code. Where this lab's measured numbers differ in
detail from the guide's printed §10 table, that is expected -- the original
script's random draws are not available -- and the README says so plainly.

Everything here is a TOY: one attention layer, random weights, tiny vectors.
It teaches mechanics (what is computed, what is stored, what survives a
channel), not capability. Real models are deeper, trained, and messier in
exactly the ways the guide's §7 and §9 describe.

Needs only Python 3 (standard library -- no downloads, no numpy).
Run:  python3 latent_lab.py        (about a second)
"""

from __future__ import annotations

import math
import random
import time

SEED = 20261031

# --------------------------------------------------------------------------
# small vector helpers (lists of floats; the lab is tiny on purpose)
# --------------------------------------------------------------------------


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def scale(a, s):
    return [x * s for x in a]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def matvec(m, v):
    return [dot(row, v) for row in m]


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    tot = sum(ex)
    return [e / tot for e in ex]


def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def rand_matrix(rng, rows, cols, sd=1.0):
    return [[rng.gauss(0.0, sd) for _ in range(cols)] for _ in range(rows)]


def rand_unit(rng, d):
    v = [rng.gauss(0.0, 1.0) for _ in range(d)]
    n = norm(v)
    return [x / n for x in v]


def orthonormal_basis(rng, d):
    """d orthonormal vectors in R^d via Gram-Schmidt on random gaussians."""
    basis = []
    for _ in range(d):
        v = [rng.gauss(0.0, 1.0) for _ in range(d)]
        for q in basis:
            v = [x - dot(v, q) * y for x, y in zip(v, q)]
        n = norm(v)
        basis.append([x / n for x in v])
    return basis


# --------------------------------------------------------------------------
# §2.6 -- how big is a KV cache?
# bytes per token = 2 (K and V) x layers x KV heads x head size x bytes/number
# --------------------------------------------------------------------------


def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_number):
    return 2 * layers * kv_heads * head_dim * bytes_per_number


def kv_cache_bytes(layers, kv_heads, head_dim, tokens, bytes_per_number):
    return kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_number) * tokens


# (layers, KV heads, head size) -- the guide's calculator presets (§2.6)
MODEL_PRESETS = {
    "GPT-2 small": (12, 12, 64),
    "Llama 3 8B": (32, 8, 128),
    "Llama 3 70B": (80, 8, 128),
}

TEXT_BYTES_PER_TOKEN = 4  # the guide's "same tokens as text (~4 bytes/token)"


def calculator_demo(tokens=1000, bytes_per_number=2):
    """Guide §2.6 worked example, computed: Llama 3 8B, 1,000 tokens, fp16."""
    layers, heads, hd = MODEL_PRESETS["Llama 3 8B"]
    per_token = kv_bytes_per_token(layers, heads, hd, bytes_per_number)
    total = per_token * tokens
    text = tokens * TEXT_BYTES_PER_TOKEN
    return {
        "model": "Llama 3 8B",
        "tokens": tokens,
        "bytes_per_token": per_token,
        "kib_per_token": per_token / 1024,
        "total_bytes": total,
        "mib_total": total / (1024 * 1024),
        "text_bytes": text,
        "ratio": total / text,
        "send_seconds_10gbit": total * 8 / 10e9,
        "send_seconds_1gbit": total * 8 / 1e9,
        "presets_bytes_per_token": {
            name: kv_bytes_per_token(l, h, d, bytes_per_number)
            for name, (l, h, d) in MODEL_PRESETS.items()
        },
    }


def channel_capacity_bits():
    """Guide §3: what one token can carry vs one hidden-state vector."""
    return {
        "token_bits": math.log2(128256),          # vocab of the guide's example
        "hidden_bits": 4096 * 16,                  # 4,096 numbers at 16 bits
    }


# --------------------------------------------------------------------------
# a tiny one-layer "model" with random weights (mechanics, not capability)
# --------------------------------------------------------------------------


class ToyModel:
    """Single attention layer over token embeddings; greedy decoding.

    K/V "rows computed" are counted honestly: every key or value vector this
    object actually computes increments the counter by one.
    """

    def __init__(self, vocab=64, d_model=16, seed=SEED):
        rng = random.Random(seed)
        self.d = d_model
        self.vocab = vocab
        self.emb = rand_matrix(rng, vocab, d_model, sd=0.5)
        self.wq = rand_matrix(rng, d_model, d_model, sd=1 / math.sqrt(d_model))
        self.wk = rand_matrix(rng, d_model, d_model, sd=1 / math.sqrt(d_model))
        self.wv = rand_matrix(rng, d_model, d_model, sd=1 / math.sqrt(d_model))
        self.wo = rand_matrix(rng, vocab, d_model, sd=1 / math.sqrt(d_model))
        self.rows_computed = 0

    def key(self, token):
        self.rows_computed += 1
        return matvec(self.wk, self.emb[token])

    def value(self, token):
        self.rows_computed += 1
        return matvec(self.wv, self.emb[token])

    def hidden_and_logits(self, token, keys, values):
        """Hidden state + next-token logits for `token` attending to keys/values
        (which must include this token's own K/V as the last entries)."""
        q = matvec(self.wq, self.emb[token])
        scores = [dot(q, k) / math.sqrt(self.d) for k in keys]
        w = softmax(scores)
        ctx = [0.0] * self.d
        for wi, v in zip(w, values):
            for i in range(self.d):
                ctx[i] += wi * v[i]
        h = add(self.emb[token], ctx)
        return h, matvec(self.wo, h)


def generate_compare(n_tokens=12, prompt_token=7):
    """Guide §10 experiment 1: KV cache vs recompute, 12 tokens.

    Both paths greedily generate the same 12-token sequence. The no-cache
    path recomputes every earlier token's K and V at every step, so its row
    count is 2 * (1 + 2 + ... + n); the cached path computes each token's
    K and V once: 2 * n. The outputs must agree to float precision.
    """
    # --- with cache
    m = ToyModel()
    cache_k, cache_v, seq, logits_cached = [], [], [prompt_token], []
    for _ in range(n_tokens):
        tok = seq[-1]
        cache_k.append(m.key(tok))
        cache_v.append(m.value(tok))
        _, logits = m.hidden_and_logits(tok, cache_k, cache_v)
        logits_cached.append(logits)
        seq.append(max(range(len(logits)), key=lambda i: logits[i]))
    rows_cached = m.rows_computed

    # --- without cache: full recompute at every step
    m2 = ToyModel()
    seq2, logits_plain = [prompt_token], []
    for _ in range(n_tokens):
        ks = [m2.key(t) for t in seq2]
        vs = [m2.value(t) for t in seq2]
        _, logits = m2.hidden_and_logits(seq2[-1], ks, vs)
        logits_plain.append(logits)
        seq2.append(max(range(len(logits)), key=lambda i: logits[i]))
    rows_plain = m2.rows_computed

    max_diff = max(abs(a - b) for la, lb in zip(logits_cached, logits_plain)
                   for a, b in zip(la, lb))
    return {
        "n_tokens": n_tokens,
        "rows_without_cache": rows_plain,
        "rows_with_cache": rows_cached,
        "tokens_identical": seq == seq2,
        "max_logit_diff": max_diff,
    }


def agent_handoff(doc_tokens=2000, question_tokens=20, seed=SEED + 1):
    """Guide §10 experiment 2 / §4.2: A reads a document, B answers a question.

    Without a handoff B must compute K/V rows for the document AND its own
    question: 2 * (2000 + 20) = 4,040 rows. With A's cache handed over, B
    computes only its question's rows: 2 * 20 = 40. B's next-token logits are
    identical either way, because they are computed from the same K/V values.
    The cache size is counted for THIS toy model (1 layer, d=16, at 16 bits).
    """
    rng = random.Random(seed)
    doc = [rng.randrange(64) for _ in range(doc_tokens)]
    question = [rng.randrange(64) for _ in range(question_tokens)]

    # Agent A prefills the document once.
    a = ToyModel()
    doc_k = [a.key(t) for t in doc]
    doc_v = [a.value(t) for t in doc]

    def b_logits(reuse_cache):
        b = ToyModel()
        if reuse_cache:
            ks, vs = list(doc_k), list(doc_v)
        else:
            ks = [b.key(t) for t in doc]
            vs = [b.value(t) for t in doc]
        for t in question:
            ks.append(b.key(t))
            vs.append(b.value(t))
        _, logits = b.hidden_and_logits(question[-1], ks, vs)
        return logits, b.rows_computed

    logits_full, rows_full = b_logits(False)
    logits_hand, rows_hand = b_logits(True)
    max_diff = max(abs(x - y) for x, y in zip(logits_full, logits_hand))
    return {
        "doc_tokens": doc_tokens,
        "question_tokens": question_tokens,
        "rows_without_handoff": rows_full,
        "rows_with_handoff": rows_hand,
        "speedup": rows_full / rows_hand,
        "max_logit_diff": max_diff,
        "cache_bytes_toy": doc_tokens * 2 * 16 * 2,  # tokens x (K,V) x d x fp16
    }


def prefix_accounting(doc=2000, own=100, message=150):
    """Guide §3 problem 2 / §4.1: three agents share a 2,000-token document.

    Text only: Planner reads doc+own (2,100); Solver adds one message
    (2,250); Checker adds two (2,400) -- total 6,750 token-prefills, of
    which 6,000 are the same document three times.
    Prefix caching: the document is prefilled once; each agent computes
    only its own prompt and the messages it receives: 2,750.
    Full latent handoff: messages also arrive as caches, so each agent
    computes only its own new tokens: 2,300.
    """
    text_only = (doc + own) + (doc + own + message) + (doc + own + 2 * message)
    prefix = doc + own + (own + message) + (own + 2 * message)
    latent = doc + 3 * own
    return {
        "text_only": text_only,
        "prefix_caching": prefix,
        "full_latent": latent,
        "prefix_saving_pct": 100 * (1 - prefix / text_only),
        "latent_saving_pct": 100 * (1 - latent / text_only),
        "shared_doc_reads_in_text_mode": 3 * doc,
    }


# --------------------------------------------------------------------------
# §3 / §5.2 -- the text bottleneck and CIPHER-style soft tokens
# The sender is 50/30/20 unsure between three answers ("8", "12", "6" --
# the guide's §10 experiment 3). Three channels carry that thought:
#   text   -- only the argmax token's embedding
#   soft   -- the probability-weighted blend of embeddings (CIPHER), with
#             temperature reshaping the sender's distribution first
#   latent -- the exact blend at the sender's true probabilities
# In this toy the latent vector and the T=1 soft token are the same point;
# in a real model the latent state is a hidden state, not an embedding
# blend, and carries context the blend does not (guide §5.1). The toy keeps
# them in one space so the temperature dial is visible in exact arithmetic:
# embeddings here are orthonormal, so every number below is computed, not
# fitted.
# --------------------------------------------------------------------------

CANDIDATES = ("8", "12", "6")
TRUE_P = (0.5, 0.3, 0.2)


def _temperature_weights(p, temp):
    w = [x ** (1.0 / temp) for x in p]
    tot = sum(w)
    return [x / tot for x in w]


def thought_channels(temps=(0.25, 0.5, 1.0, 2.0, 4.0), seed=SEED + 2):
    rng = random.Random(seed)
    basis = orthonormal_basis(rng, 50)
    emb = {name: basis[i] for i, name in enumerate(CANDIDATES)}
    thought = [sum(TRUE_P[i] * emb[name][j] for i, name in enumerate(CANDIDATES))
               for j in range(50)]

    def describe(vec):
        # receiver recovers a distribution by projecting onto the candidates
        raw = [max(0.0, dot(vec, emb[name])) for name in CANDIDATES]
        tot = sum(raw)
        q = [x / tot for x in raw] if tot > 0 else [0.0] * 3
        return {
            "cosine_to_thought": cosine(vec, thought),
            "recovered": q,
            "doubt_preserved": q[1] + q[2],   # true doubt is 0.3 + 0.2 = 0.5
        }

    out = {"true_p": list(TRUE_P), "channels": {}}
    out["channels"]["text"] = describe(emb["8"])
    for t in temps:
        w = _temperature_weights(TRUE_P, t)
        vec = [sum(w[i] * emb[name][j] for i, name in enumerate(CANDIDATES))
               for j in range(50)]
        d = describe(vec)
        d["weights"] = w
        out["channels"][f"soft_T{t}"] = d
    out["channels"]["latent"] = describe(thought)
    out["channels"]["latent"]["weights"] = list(TRUE_P)
    return out


def candidates_recovered(trials=400, d=32, vocab=24, seed=SEED + 3):
    """How much of the sender's candidate set survives each channel?

    Random (non-orthogonal) unit embeddings this time -- the realistic,
    messy case. The receiver ranks the whole vocabulary by projection onto
    the received vector; we report the expected fraction of the 3 true
    candidates that land in the receiver's top 3, per channel.
    """
    rng = random.Random(seed)
    found = {"text": 0.0, "soft_T0.5": 0.0, "latent": 0.0}
    for _ in range(trials):
        emb = [rand_unit(rng, d) for _ in range(vocab)]
        cand = (0, 1, 2)
        thought = [sum(TRUE_P[i] * emb[c][j] for i, c in enumerate(cand))
                   for j in range(d)]
        w_soft = _temperature_weights(TRUE_P, 0.5)
        soft = [sum(w_soft[i] * emb[c][j] for i, c in enumerate(cand))
                for j in range(d)]
        chans = {"text": emb[0], "soft_T0.5": soft, "latent": thought}
        for name, vec in chans.items():
            u = scale(vec, 1.0 / norm(vec))
            scores = sorted(range(vocab), key=lambda j: -dot(u, emb[j]))
            found[name] += len(set(scores[:3]) & set(cand)) / 3.0
    return {name: 100.0 * v / trials for name, v in found.items()}


# --------------------------------------------------------------------------
# §5 -- thought-vector handoff on a toy task with a right answer
# Agent A is unsure how many pens Maya buys: 4 (50%), 6 (30%) or 2 (20%).
# Pens cost $3; she pays with $20. Agent B must state the EXPECTED change
# and the probability the cost exceeds $15. Both are functions of A's whole
# distribution. Text ("4") destroys them; the vector keeps them.
# --------------------------------------------------------------------------


def thought_handoff(seed=SEED + 2):
    rng = random.Random(seed)
    basis = orthonormal_basis(rng, 50)
    quantities = (4, 6, 2)
    thought = [sum(TRUE_P[i] * basis[i][j] for i in range(3)) for j in range(50)]

    def decode(vec):
        raw = [max(0.0, dot(vec, basis[i])) for i in range(3)]
        tot = sum(raw)
        return [x / tot for x in raw]

    p_latent = decode(thought)          # exact recovery (orthonormal toy)
    p_text = [1.0, 0.0, 0.0]            # B only ever sees the word "4"

    def expected_change(p):
        eq = sum(pi * q for pi, q in zip(p, quantities))
        return 20.0 - 3.0 * eq

    def prob_cost_over_15(p):
        return sum(pi for pi, q in zip(p, quantities) if 3 * q > 15)

    true_ec = expected_change(TRUE_P)
    return {
        "quantities": list(quantities),
        "true_p": list(TRUE_P),
        "expected_change_true": true_ec,
        "text": {
            "expected_change": expected_change(p_text),
            "error": abs(expected_change(p_text) - true_ec),
            "prob_cost_over_15": prob_cost_over_15(p_text),
        },
        "latent": {
            "expected_change": expected_change(p_latent),
            "error": abs(expected_change(p_latent) - true_ec),
            "prob_cost_over_15": prob_cost_over_15(p_latent),
        },
        "prob_cost_over_15_true": prob_cost_over_15(TRUE_P),
    }


# --------------------------------------------------------------------------
# §7.2 / §10 experiment 4 -- different models need a translator
# Model Y's space is model X's space rotated by an unknown orthogonal map.
# A linear adapter is fitted by ridge least squares from n noisy anchor
# pairs (the same input seen by both models). With far fewer anchors than
# dimensions the fit only covers a subspace; with about twice as many it is
# near-perfect. Real model pairs are NOT related by a rotation -- the guide
# is explicit that practical systems train neural fusers -- so read this as
# the shape of the problem, not its difficulty.
# --------------------------------------------------------------------------


def _solve(a, b):
    """Solve a x = b (a: n x n, b: n x m) by Gaussian elimination."""
    n = len(a)
    m = len(b[0])
    aug = [row[:] + brow[:] for row, brow in zip(a, b)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [x / pv for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return [row[n:n + m] for row in aug]


def cross_model_adapter(anchor_counts=(5, 10, 16, 30, 50), d=16,
                        noise=0.03, trials=200, seed=SEED + 4):
    rng = random.Random(seed)
    basis = orthonormal_basis(rng, d)          # the unknown rotation R
    def rot(v):
        return [dot(row, v) for row in basis]
    test = [rand_unit(rng, d) for _ in range(trials)]
    unchanged = sum(cosine(v, rot(v)) for v in test) / trials

    results = {"d": d, "unchanged_similarity": unchanged, "by_anchors": {}}
    max_n = max(anchor_counts)
    anchors_x = [rand_unit(rng, d) for _ in range(max_n)]
    anchors_y = [[x + rng.gauss(0.0, noise) for x in rot(v)] for v in anchors_x]
    for n in anchor_counts:
        ax, ay = anchors_x[:n], anchors_y[:n]
        # ridge least squares: M = cross @ gram^-1 with
        # gram = Ax Ax^T (+ lam I), cross = Ay Ax^T. Solved as
        # gram @ Y = cross^T  ->  Y = M^T  (gram is symmetric).
        gram = [[sum(ax[k][i] * ax[k][j] for k in range(n)) + (0.01 if i == j else 0.0)
                 for j in range(d)] for i in range(d)]
        cross_t = [[sum(ay[k][i] * ax[k][j] for k in range(n))
                    for i in range(d)] for j in range(d)]
        mt = _solve(gram, cross_t)             # mt[j][i] = M[i][j]
        def mapped(v):
            return [sum(mt[j][i] * v[j] for j in range(d)) for i in range(d)]
        sim = sum(cosine(mapped(v), rot(v)) for v in test) / trials
        results["by_anchors"][n] = sim
    return results


# --------------------------------------------------------------------------
# §4.3 / §10 experiment 5 -- positions (RoPE): reuse is exact after
# re-rotation. A's keys were cached at positions 0..99; B needs them at
# 50..149. Using them as-is scrambles attention; rotating every cached key
# by +50 steps reproduces the true keys exactly.
# --------------------------------------------------------------------------


def _rope_rotate(vec, pos, thetas):
    out = list(vec)
    for i, th in enumerate(thetas):
        a = pos * th
        c, s = math.cos(a), math.sin(a)
        x, y = vec[2 * i], vec[2 * i + 1]
        out[2 * i] = x * c - y * s
        out[2 * i + 1] = x * s + y * c
    return out


def rope_repair(n_tokens=100, shift=50, d_head=8, seed=SEED + 5, fix_shift=None):
    if fix_shift is None:
        fix_shift = shift
    rng = random.Random(seed)
    thetas = [10000 ** (-2 * i / d_head) for i in range(d_head // 2)]
    base_keys = [[rng.gauss(0.0, 1.0) for _ in range(d_head)]
                 for _ in range(n_tokens)]
    q_base = [rng.gauss(0.0, 1.0) for _ in range(d_head)]
    q = _rope_rotate(q_base, shift + n_tokens, thetas)   # B's next position

    def attention(keys):
        scores = [dot(q, k) / math.sqrt(d_head) for k in keys]
        return softmax(scores)

    true_keys = [_rope_rotate(k, shift + j, thetas)
                 for j, k in enumerate(base_keys)]
    cached_keys = [_rope_rotate(k, j, thetas) for j, k in enumerate(base_keys)]
    fixed_keys = [_rope_rotate(k, fix_shift, thetas) for k in cached_keys]

    p_true = attention(true_keys)
    p_asis = attention(cached_keys)
    p_fixed = attention(fixed_keys)
    tv = lambda p, r: 50.0 * sum(abs(a - b) for a, b in zip(p, r))  # in %
    key_err = max(abs(a - b) for ka, kb in zip(fixed_keys, true_keys)
                  for a, b in zip(ka, kb))
    return {
        "n_tokens": n_tokens,
        "shift": shift,
        "attention_misplaced_pct_as_is": tv(p_asis, p_true),
        "attention_misplaced_pct_fixed": tv(p_fixed, p_true),
        "max_key_error_after_fix": key_err,
    }


# --------------------------------------------------------------------------
# §6 -- one problem, three ways (Planner -> Solver), all figures computed
# from the functions above with the guide's Llama-3-8B-class dimensions.
# --------------------------------------------------------------------------


def three_ways(plan_tokens=20, doc_tokens=2000, latent_steps=8):
    layers, heads, hd = MODEL_PRESETS["Llama 3 8B"]
    per_token = kv_bytes_per_token(layers, heads, hd, 2)
    return {
        "text_bytes": plan_tokens * TEXT_BYTES_PER_TOKEN,
        "kv_bytes": plan_tokens * per_token,
        "thought_bytes": latent_steps * 4096 * 2,
        "doc_kv_bytes": doc_tokens * per_token,
        "doc_text_bytes": doc_tokens * TEXT_BYTES_PER_TOKEN,
        "plan_tokens": plan_tokens,
        "latent_steps": latent_steps,
    }


def all_numbers():
    """Every headline number in this lab, computed. (Used for the episode's
    demo segment and by validate.py to keep the README honest.)"""
    return {
        "calculator_llama3_8b_1000tok_fp16": calculator_demo(),
        "channel_capacity_bits": channel_capacity_bits(),
        "exp1_cache_vs_recompute": generate_compare(),
        "exp2_agent_handoff": agent_handoff(),
        "prefix_accounting": prefix_accounting(),
        "thought_channels": thought_channels(),
        "candidates_recovered_pct": candidates_recovered(),
        "thought_handoff": thought_handoff(),
        "cross_model_adapter": cross_model_adapter(),
        "rope_repair": rope_repair(),
        "section6_three_ways": three_ways(),
    }


def _fmt_bytes(n):
    if n >= 1024 * 1024:
        return f"{n:,} bytes ({n / (1024 * 1024):.4g} MiB)"
    if n >= 1024:
        return f"{n:,} bytes ({n / 1024:.4g} KiB)"
    return f"{n:,} bytes"


def main():
    t0 = time.time()
    print("=" * 74)
    print("LATENT LAB -- how AI agents can talk without words")
    print("toy models, real arithmetic. Guide sections in brackets.")
    print("=" * 74)

    c = calculator_demo()
    bits = channel_capacity_bits()
    print("\n[§2.6] How big is a KV cache?  Llama 3 8B, 1,000 tokens, 16-bit")
    print(f"  bytes per token = 2 x 32 layers x 8 heads x 128 x 2 B"
          f" = {c['bytes_per_token']:,}  ({c['kib_per_token']:.0f} KiB)")
    print(f"  cache for 1,000 tokens : {_fmt_bytes(c['total_bytes'])}")
    print(f"  same tokens as text    : {_fmt_bytes(c['text_bytes'])}"
          f"  -> cache is {c['ratio']:,.0f}x bigger")
    print(f"  sending it takes {c['send_seconds_10gbit']:.3f} s at 10 Gb/s,"
          f" {c['send_seconds_1gbit']:.2f} s at 1 Gb/s")
    print(f"  presets (bytes/token, fp16): " +
          ", ".join(f"{k} {v:,}" for k, v in c["presets_bytes_per_token"].items()))
    print(f"[§3] one token carries at most {bits['token_bits']:.1f} bits;"
          f" one hidden vector has {bits['hidden_bits']:,} bits of raw capacity")

    e1 = generate_compare()
    print(f"\n[§10 exp 1] KV cache vs recompute, {e1['n_tokens']} tokens")
    print(f"  key/value rows computed: {e1['rows_without_cache']} -> "
          f"{e1['rows_with_cache']}; generated tokens identical: "
          f"{e1['tokens_identical']}; max logit difference "
          f"{e1['max_logit_diff']:.2e}")

    e2 = agent_handoff()
    pa = prefix_accounting()
    print(f"\n[§10 exp 2 / §4] Agent handoff: 2,000-token document + 20-token question")
    print(f"  B's key/value rows: {e2['rows_without_handoff']:,} -> "
          f"{e2['rows_with_handoff']}  ({e2['speedup']:.0f}x less);"
          f" logits identical (max diff {e2['max_logit_diff']:.2e})")
    print(f"  the handed-over cache, in this toy (d=16, 1 layer, fp16): "
          f"{e2['cache_bytes_toy']:,} bytes")
    print(f"  three agents sharing the document -- tokens prefilled:")
    print(f"    text only {pa['text_only']:,}  |  prefix caching "
          f"{pa['prefix_caching']:,} ({pa['prefix_saving_pct']:.0f}% less)  |  "
          f"full latent {pa['full_latent']:,} ({pa['latent_saving_pct']:.0f}% less)")

    tc = thought_channels()
    rec = candidates_recovered()
    print(f"\n[§10 exp 3 / §5.2] The text bottleneck: a thought that is "
          f"50/30/20 between three answers")
    print(f"  {'channel':<12} {'cos to thought':>14} {'doubt kept':>11}")
    for name, d in tc["channels"].items():
        print(f"  {name:<12} {d['cosine_to_thought']:>14.3f} "
              f"{d['doubt_preserved']:>11.3f}")
    print(f"  candidates found in the receiver's top 3 (random embeddings,"
          f" % of the 3):")
    for name, v in rec.items():
        print(f"    {name:<12} {v:.0f}%")

    th = thought_handoff()
    print(f"\n[§5] Thought-vector handoff: expected change from $20"
          f" (pens at $3, quantity unsure)")
    print(f"  true expected change ${th['expected_change_true']:.2f};"
          f" P(cost over $15) = {th['prob_cost_over_15_true']:.2f}")
    print(f"  via text   : change ${th['text']['expected_change']:.2f}"
          f" (error ${th['text']['error']:.2f}),"
          f" P(over $15) = {th['text']['prob_cost_over_15']:.2f}")
    print(f"  via latent : change ${th['latent']['expected_change']:.2f}"
          f" (error ${th['latent']['error']:.2f}),"
          f" P(over $15) = {th['latent']['prob_cost_over_15']:.2f}")

    ad = cross_model_adapter()
    print(f"\n[§10 exp 4 / §7.2] Different models: similarity of a transferred"
          f" thought (d={ad['d']})")
    print(f"  sent unchanged: {ad['unchanged_similarity']:+.3f}")
    for n, s in ad["by_anchors"].items():
        print(f"  adapter from {n:>2} anchor pairs: {s:.3f}")

    rp = rope_repair()
    print(f"\n[§10 exp 5 / §4.3] Positions (RoPE): cache written at 0-99,"
          f" needed at 50-149")
    print(f"  attention misplaced using the cache as-is: "
          f"{rp['attention_misplaced_pct_as_is']:.1f}%")
    print(f"  after re-rotating every key by +50 steps  : "
          f"{rp['attention_misplaced_pct_fixed']:.4f}%"
          f"  (max key error {rp['max_key_error_after_fix']:.2e})")

    tw = three_ways()
    print(f"\n[§6] One problem, three ways -- what the Planner sends the Solver")
    print(f"  text          : {_fmt_bytes(tw['text_bytes'])}"
          f"  (Solver must re-read {tw['plan_tokens']} tokens)")
    print(f"  KV cache      : {_fmt_bytes(tw['kv_bytes'])}"
          f"  (Solver computes nothing for those tokens)")
    print(f"  thought vector: {_fmt_bytes(tw['thought_bytes'])}"
          f"  ({tw['latent_steps']} latent steps x 4,096 numbers x 2 B)")
    print(f"  if the 2,000-token document travels too: KV "
          f"{_fmt_bytes(tw['doc_kv_bytes'])} vs text "
          f"{_fmt_bytes(tw['doc_text_bytes'])}")

    print(f"\nTotal runtime: {time.time() - t0:.2f} s (toy scale -- mechanics,"
          f" not capability)")


if __name__ == "__main__":
    main()
