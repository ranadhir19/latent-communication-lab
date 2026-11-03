# latent-communication-lab

**How AI agents can talk without words** — a hands-on lab in latent
communication: passing KV caches and continuous "thought vectors" between
LLM agents instead of text.

This is the training-session companion code for The Architect's AI Workshop
episode *"How AI Agents Can Talk Without Words"* and for the written guide
of the same name. The guide's §10 references a `latent_lab.py` that was
never published with it; this repository is an independent rebuild of that
lab from the guide's text — same five experiments, same toy scale, every
number computed fresh by the code here.

## Run it

```bash
python3 latent_lab.py
```

One command, a few seconds, **no downloads** — pure Python 3 standard
library (not even numpy). Tests and the full validation harness:

```bash
python3 -m unittest test_latent_lab   # 20 tests
python3 validate.py                   # compile + tests + consistency + scan
```

## What it shows (this lab's measured results)

| Experiment | Result |
|---|---|
| KV cache size, Llama 3 8B, 1,000 tokens, fp16 | 131,072 bytes/token → **125 MiB** total; the same tokens as text ≈ 4,000 bytes (**32,768×** less); sending the cache takes 0.105 s at 10 Gb/s |
| Cache vs recompute (12 tokens) | **156 → 24** key/value rows computed; outputs bit-identical (max logit difference 0) |
| Agent handoff (2,000-token doc + 20-token question) | receiver computes **4,040 → 40** rows (**101×** less), identical output; the cache in this toy (d = 16, 1 layer) is 128,000 bytes |
| Three agents sharing a document | tokens prefilled: **6,750** (text) → **2,750** (prefix caching, 59% less) → **2,300** (full latent handoff, 66% less) |
| Text bottleneck (a 50/30/20 thought) | cosine to the true thought: text **0.811**, soft token (T = 0.5) **0.966**, latent **1.000**; doubt preserved: **0.000 / 0.342 / 0.500**; candidates recovered in the receiver's top 3: **39% / 63% / 80%** |
| Thought-vector handoff (expected-value task) | expected change **$7.40** via latent (error $0.00) vs **$8.00** via text (error $0.60); P(cost over $15): 0.30 via latent, 0.00 via text |
| Different models (linear adapter) | similarity sent unchanged **−0.108**; adapter from 5 / 10 / 16 / 30 / 50 anchor pairs: **0.52 / 0.76 / 0.94 / 0.995 / 0.997** |
| Positions (RoPE) | attention misplaced using a shifted cache as-is: **48.1%**; after re-rotating every key: **0.0000%** (exact) |
| One problem, three ways (§6) | what the Planner sends: text **80 bytes**, KV cache **2.5 MiB**, thought vectors **64 KiB**; the shared 2,000-token document would travel as **250 MiB** of cache vs ~8 KB of text |

## Map: lab ↔ guide ↔ video

| Lab section | Guide section | Idea |
|---|---|---|
| KV-cache calculator + presets | §2.6 "How big is a KV cache?" | `2 × layers × KV heads × head size × bytes` per token; cache vs text size; bandwidth cost |
| Cache vs recompute | §2.6, §10 exp. 1 | caching is *exact*, and work grows linearly instead of quadratically |
| Agent handoff + prefix accounting | §4.1–4.2, §3 problem 2, §10 exp. 2 | prefix caching and cache handoff remove repeated prefill |
| Text bottleneck + soft tokens | §3, §5.2 (CIPHER), §10 exp. 3 | argmax throws the doubt away; temperature is a dial |
| Thought-vector handoff | §5.1, §5.3 | a downstream agent can compute with the sender's whole distribution |
| Cross-model adapter | §4.4, §7.2, §10 exp. 4 | different models speak different vector languages; translators need anchors ≫ dimensions |
| RoPE repair | §4.3, §10 exp. 5 | positions are exactly fixable by re-rotation; *context* mismatch is the harder part |
| One problem, three ways | §6 | text / KV / thought-vector channels side by side |

## Honest boundaries — read this before citing the numbers

- **Toy scale.** The "models" here are one attention layer with random
  weights and 16-number vectors. The lab teaches *mechanics* — what is
  computed, stored, and lost — not capability. Nothing here shows a real
  model doing anything.
- **A rebuild, not the original.** The guide's printed §10 table came from
  the author's original script, which was not published. This lab's numbers
  are its own measurements: most land close to the guide's (101× handoff,
  0.81 text cosine, adapter 0.52→0.997), and two differ visibly — the
  guide's toy put 1.9% of attention in the wrong place for the RoPE case
  (this toy: 48.1%; the *fix* is exact in both), and its unregularised
  16-anchor adapter dipped to 0.67 where this lab's ridge-regularised fit
  reaches 0.94. The mechanisms and the orderings are the content; treat
  any single decimal as this implementation's, not a constant of nature.
- **Cross-model transfer is not solved here.** The adapter experiment maps
  between two spaces related by a pure *rotation*, fitted exactly. Real
  model pairs are not rotations of each other; practical systems train
  neural fusers and validate every pair (guide §7.2, §9). Likewise, KV
  handoff in this lab assumes the same model, tokenizer, and positions —
  the guide's compatibility checklist (§7.1) is the real-world fine print.
- **Text keeps its virtues.** Latent channels give up auditability and
  universality for bandwidth and speed. The guide's §3 insight stands:
  text remains the right interface at system boundaries.

## Files

- `latent_lab.py` — the lab (all experiments, one file, stdlib only)
- `test_latent_lab.py` — 20 tests of the numerical claims, including
  negative tests (shuffled anchors and wrong RoPE shifts must fail)
- `validate.py` — validation harness: compile/import, full test suite,
  README-vs-recomputation consistency, forbidden-reference scan
