#!/usr/bin/env python3
"""Tests for latent_lab.py -- they test the lab's numerical claims, not its
code paths only. Every expected value below is either the guide's published
arithmetic (re-derived independently in the test) or a measured property of
the lab's own construction (direction / ordering / exactness). Negative
tests are included: shuffled adapters and wrong RoPE shifts MUST fail, so
the passing results cannot be tautologies.

Run: python3 -m unittest test_latent_lab -v   (or: python3 validate.py)
"""

import math
import unittest

import latent_lab as lab


class TestKVCacheSize(unittest.TestCase):
    def test_llama3_8b_guide_arithmetic(self):
        # guide §2.6: 2 x 32 x 8 x 128 x 2 = 131,072 B/token; x1000 = 125 MiB
        per = lab.kv_bytes_per_token(32, 8, 128, 2)
        self.assertEqual(per, 2 * 32 * 8 * 128 * 2)
        self.assertEqual(per, 131_072)
        total = lab.kv_cache_bytes(32, 8, 128, 1000, 2)
        self.assertEqual(total, 131_072_000)
        self.assertAlmostEqual(total / (1024 * 1024), 125.0)

    def test_presets(self):
        c = lab.calculator_demo()
        self.assertEqual(c["presets_bytes_per_token"]["GPT-2 small"], 36_864)
        self.assertEqual(c["presets_bytes_per_token"]["Llama 3 70B"], 327_680)
        self.assertEqual(c["bytes_per_token"], 131_072)
        self.assertEqual(c["kib_per_token"], 128.0)

    def test_ratio_and_send_time(self):
        c = lab.calculator_demo()
        # text at the guide's ~4 bytes/token: 131,072,000 / 4,000 = 32,768
        self.assertEqual(c["ratio"], 32_768.0)
        self.assertAlmostEqual(c["send_seconds_10gbit"], 0.1048576, places=6)
        self.assertAlmostEqual(c["send_seconds_1gbit"], 1.048576, places=6)

    def test_precision_scales_bytes(self):
        # near-miss: the same model at fp32 must cost exactly 2x fp16
        self.assertEqual(lab.kv_bytes_per_token(32, 8, 128, 4),
                         2 * lab.kv_bytes_per_token(32, 8, 128, 2))

    def test_channel_capacity_bits(self):
        b = lab.channel_capacity_bits()
        self.assertTrue(16.9 < b["token_bits"] < 17.1)   # guide: ~17 bits
        self.assertEqual(b["hidden_bits"], 65_536)       # 4,096 x 16


class TestCacheVsRecompute(unittest.TestCase):
    def test_guide_counts_and_exactness(self):
        e = lab.generate_compare()
        self.assertEqual(e["rows_without_cache"], 156)   # 2*(1+..+12)
        self.assertEqual(e["rows_with_cache"], 24)       # 2*12
        self.assertTrue(e["tokens_identical"])
        self.assertLess(e["max_logit_diff"], 1e-9)

    def test_count_formula_at_another_length(self):
        e = lab.generate_compare(n_tokens=5)
        self.assertEqual(e["rows_without_cache"], 2 * (1 + 2 + 3 + 4 + 5))
        self.assertEqual(e["rows_with_cache"], 2 * 5)


class TestHandoffAndPrefix(unittest.TestCase):
    def test_handoff_counts(self):
        e = lab.agent_handoff()
        self.assertEqual(e["rows_without_handoff"], 4_040)
        self.assertEqual(e["rows_with_handoff"], 40)
        self.assertAlmostEqual(e["speedup"], 101.0)
        self.assertLess(e["max_logit_diff"], 1e-9)
        self.assertEqual(e["cache_bytes_toy"], 128_000)  # guide §10 exp 2

    def test_prefix_accounting(self):
        pa = lab.prefix_accounting()
        self.assertEqual(pa["text_only"], 6_750)         # guide §4.1 table
        self.assertEqual(pa["prefix_caching"], 2_750)
        self.assertEqual(pa["full_latent"], 2_300)
        self.assertAlmostEqual(pa["prefix_saving_pct"], 59.26, places=1)
        self.assertAlmostEqual(pa["latent_saving_pct"], 65.93, places=1)
        self.assertEqual(pa["shared_doc_reads_in_text_mode"], 6_000)


class TestThoughtChannels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tc = lab.thought_channels()

    def test_cosine_ordering(self):
        ch = self.tc["channels"]
        self.assertAlmostEqual(ch["latent"]["cosine_to_thought"], 1.0, places=9)
        self.assertAlmostEqual(ch["text"]["cosine_to_thought"], 0.811, places=2)
        self.assertGreater(ch["soft_T0.5"]["cosine_to_thought"],
                           ch["text"]["cosine_to_thought"])

    def test_doubt(self):
        ch = self.tc["channels"]
        self.assertAlmostEqual(ch["text"]["doubt_preserved"], 0.0, places=9)
        self.assertAlmostEqual(ch["latent"]["doubt_preserved"], 0.5, places=9)
        self.assertTrue(0.30 < ch["soft_T0.5"]["doubt_preserved"] < 0.38)

    def test_temperature_is_a_dial(self):
        ch = self.tc["channels"]
        # hotter keeps more doubt but (past T=1) blurs the direction
        self.assertLess(ch["soft_T0.25"]["doubt_preserved"],
                        ch["soft_T1.0"]["doubt_preserved"])
        self.assertLess(ch["soft_T1.0"]["doubt_preserved"],
                        ch["soft_T4.0"]["doubt_preserved"])
        self.assertLess(ch["soft_T0.25"]["cosine_to_thought"],
                        ch["soft_T1.0"]["cosine_to_thought"])
        self.assertLess(ch["soft_T4.0"]["cosine_to_thought"],
                        ch["soft_T1.0"]["cosine_to_thought"])

    def test_candidates_recovered_direction(self):
        rec = lab.candidates_recovered()
        self.assertGreater(rec["latent"], rec["soft_T0.5"])
        self.assertGreater(rec["soft_T0.5"], rec["text"])
        self.assertGreater(rec["latent"], 70.0)
        self.assertLess(rec["text"], 50.0)


class TestThoughtHandoff(unittest.TestCase):
    def test_expected_value_task(self):
        th = lab.thought_handoff()
        self.assertAlmostEqual(th["expected_change_true"], 7.40, places=9)
        self.assertAlmostEqual(th["text"]["expected_change"], 8.00, places=9)
        self.assertAlmostEqual(th["text"]["error"], 0.60, places=9)
        self.assertLess(th["latent"]["error"], 1e-9)
        self.assertEqual(th["text"]["prob_cost_over_15"], 0.0)
        self.assertAlmostEqual(th["latent"]["prob_cost_over_15"], 0.30, places=9)


class TestCrossModelAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ad = lab.cross_model_adapter()

    def test_unchanged_transfer_fails(self):
        self.assertLess(abs(self.ad["unchanged_similarity"]), 0.2)

    def test_more_anchors_help(self):
        by = self.ad["by_anchors"]
        self.assertTrue(0.4 < by[5] < 0.65)          # guide: 0.55
        self.assertGreater(by[10], by[5])            # guide: 0.77
        self.assertGreater(by[30], 0.98)             # guide: 0.998
        self.assertGreater(by[50], by[10])

    def test_shuffled_anchors_must_fail(self):
        # negative test: destroy the pairing and the adapter must NOT work,
        # proving the positive result comes from the anchors, not the solver
        import random as _r
        rng = _r.Random(99)
        basis = lab.orthonormal_basis(rng, 16)
        def rot(v):
            return [lab.dot(row, v) for row in basis]
        ax = [lab.rand_unit(rng, 16) for _ in range(50)]
        ay = [rot(v) for v in ax]
        rng.shuffle(ay)
        gram = [[sum(ax[k][i] * ax[k][j] for k in range(50))
                 + (0.01 if i == j else 0.0) for j in range(16)] for i in range(16)]
        cross_t = [[sum(ay[k][i] * ax[k][j] for k in range(50))
                    for i in range(16)] for j in range(16)]
        mt = lab._solve(gram, cross_t)
        test = [lab.rand_unit(rng, 16) for _ in range(50)]
        sim = sum(lab.cosine([sum(mt[j][i] * v[j] for j in range(16))
                              for i in range(16)], rot(v)) for v in test) / 50
        self.assertLess(sim, 0.3)


class TestRopeRepair(unittest.TestCase):
    def test_rerotation_is_exact(self):
        rp = lab.rope_repair()
        self.assertGreater(rp["attention_misplaced_pct_as_is"], 20.0)
        self.assertLess(rp["attention_misplaced_pct_fixed"], 1e-3)
        self.assertLess(rp["max_key_error_after_fix"], 1e-9)

    def test_wrong_shift_must_not_fix(self):
        rp = lab.rope_repair(fix_shift=30)   # needed +50
        self.assertGreater(rp["attention_misplaced_pct_fixed"], 1.0)


class TestThreeWays(unittest.TestCase):
    def test_section6_figures(self):
        tw = lab.three_ways()
        self.assertEqual(tw["text_bytes"], 80)             # ~20 tokens as text
        self.assertEqual(tw["kv_bytes"], 2_621_440)        # 2.5 MiB
        self.assertEqual(tw["thought_bytes"], 65_536)      # 64 KiB
        self.assertEqual(tw["doc_kv_bytes"], 262_144_000)  # 250 MiB
        self.assertEqual(tw["doc_text_bytes"], 8_000)      # ~8 KB


if __name__ == "__main__":
    unittest.main()
