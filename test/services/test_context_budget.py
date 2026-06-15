import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.context_budget import (
    BudgetConfig,
    TokenBudgetTracker,
    TokenUsage,
    compress_prompt,
    global_budget,
    sliding_window_context,
)


class TestTokenUsage(unittest.TestCase):
    def test_total_is_sum(self):
        u = TokenUsage(prompt_tokens=10, completion_tokens=5)
        self.assertEqual(u.total, 15)

    def test_total_defaults_zero(self):
        u = TokenUsage()
        self.assertEqual(u.total, 0)


class TestBudgetConfig(unittest.TestCase):
    def test_defaults(self):
        c = BudgetConfig()
        self.assertEqual(c.max_total_tokens, 100_000)
        self.assertEqual(c.max_prompt_tokens, 80_000)
        self.assertEqual(c.max_completion_tokens, 20_000)
        self.assertEqual(c.warning_threshold, 0.8)

    def test_from_dict(self):
        c = BudgetConfig.from_dict({"max_total_tokens": 50_000, "warning_threshold": 0.9})
        self.assertEqual(c.max_total_tokens, 50_000)
        self.assertEqual(c.warning_threshold, 0.9)
        self.assertEqual(c.max_prompt_tokens, 80_000)


class TestTokenBudgetTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = TokenBudgetTracker(
            config=BudgetConfig(max_total_tokens=1000, warning_threshold=0.5)
        )

    def test_record_increments(self):
        self.tracker.record("openai", "hello world", "goodbye")
        usage = self.tracker._usage["openai"]
        self.assertGreater(usage.prompt_tokens, 0)
        self.assertGreater(usage.completion_tokens, 0)

    def test_summary_returns_provider_string(self):
        self.tracker.record("openai", "a" * 40)
        s = self.tracker.summary("openai")
        self.assertIn("[openai]", s)
        self.assertIn("prompt=", s)
        self.assertIn("total=", s)

    def test_summary_all_providers(self):
        self.tracker.record("a", "hello")
        self.tracker.record("b", "world")
        s = self.tracker.summary()
        self.assertIn("[a]", s)
        self.assertIn("[b]", s)

    def test_can_proceed_within_budget(self):
        self.tracker.record("openai", "hi")
        self.assertTrue(self.tracker.can_proceed("openai"))

    def test_can_proceed_exhausted(self):
        self.tracker._usage["openai"].prompt_tokens = 1000
        self.assertFalse(self.tracker.can_proceed("openai"))

    def test_can_proceed_estimated_exhausted(self):
        self.tracker._usage["openai"].prompt_tokens = 900
        self.assertFalse(self.tracker.can_proceed("openai", estimated_prompt=200))

    def test_can_proceed_unknown_provider_defaults_to_zero(self):
        self.assertTrue(self.tracker.can_proceed("unknown"))

    def test_reset_clears_usage(self):
        self.tracker.record("openai", "a" * 40)
        self.tracker.reset()
        self.assertEqual(len(self.tracker._usage), 0)


class TestCompressPrompt(unittest.TestCase):
    def test_short_prompt_unchanged(self):
        text = "hello world"
        self.assertEqual(compress_prompt(text, max_chars=100), text)

    def test_long_prompt_truncated(self):
        text = "A" * 500 + "B" * 500
        result = compress_prompt(text, max_chars=200)
        self.assertLessEqual(len(result), 200)

    def test_contains_truncation_marker(self):
        text = "## System\ninfo\n## Context\n" + "C" * 3000
        result = compress_prompt(text, max_chars=200)
        self.assertIn("[content truncated due to length]", result)
        self.assertIn("## System", result)

    def test_context_marker_respected(self):
        header = "## Intro\nsome intro text\n"
        body = "C" * 2000
        text = header + "## Context\n" + body
        result = compress_prompt(text, max_chars=200)
        self.assertIn(header, result)
        self.assertIn("[content truncated due to length]", result)

    def test_no_context_marker_uses_default_split(self):
        text = "D" * 4000
        result = compress_prompt(text, max_chars=200)
        self.assertLessEqual(len(result), 200)


class TestSlidingWindowContext(unittest.TestCase):
    def test_within_budget_returns_all(self):
        ctxs = ["hello", "world"]
        self.assertEqual(sliding_window_context(ctxs, 100), ctxs)

    def test_over_budget_truncates_oldest(self):
        ctxs = ["A" * 200, "B" * 50]
        result = sliding_window_context(ctxs, 150)
        self.assertEqual(len(result), 2)
        self.assertLessEqual(sum(len(c) for c in result), 150)
        self.assertIn("B", result[-1])

    def test_empty_list(self):
        self.assertEqual(sliding_window_context([], 100), [])

    def test_single_context_over_budget(self):
        ctxs = ["A" * 200]
        result = sliding_window_context(ctxs, 50)
        self.assertEqual(len(result), 1)
        self.assertLessEqual(len(result[0]), 50)


class TestGlobalBudget(unittest.TestCase):
    def test_global_budget_is_singleton(self):
        self.assertIsInstance(global_budget, TokenBudgetTracker)

    def test_global_budget_default_config(self):
        self.assertEqual(global_budget.config.max_total_tokens, 100_000)


if __name__ == "__main__":
    unittest.main()
