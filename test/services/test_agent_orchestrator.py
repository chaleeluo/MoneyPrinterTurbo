import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import llm
from app.services.agent_orchestrator import (
    AgentChain,
    AgentMessage,
    run_keyword_strategist,
    run_pipeline,
    run_writer_reviewer_loop,
)


class TestAgentMessage(unittest.TestCase):
    def test_dataclass_defaults(self):
        msg = AgentMessage(sender="A", receiver="B", content="hello")
        self.assertEqual(msg.sender, "A")
        self.assertEqual(msg.receiver, "B")
        self.assertEqual(msg.content, "hello")
        self.assertIsInstance(msg.timestamp, float)


class TestAgentChain(unittest.TestCase):
    def setUp(self):
        self.chain = AgentChain(task_id="test-1")

    def test_log_appends_message(self):
        self.chain.log("Writer", "Reviewer", "draft")
        self.assertEqual(len(self.chain.messages), 1)
        msg = self.chain.messages[0]
        self.assertEqual(msg.sender, "Writer")
        self.assertEqual(msg.receiver, "Reviewer")
        self.assertEqual(msg.content, "draft")

    def test_summary_includes_task_id(self):
        self.assertIn("test-1", self.chain.summary())

    def test_summary_counts_messages(self):
        self.chain.log("A", "B", "msg1")
        self.chain.log("B", "C", "msg2")
        self.assertIn("messages=2", self.chain.summary())

    def test_multiple_logs(self):
        self.chain.log("A", "B", "x")
        self.chain.log("C", "D", "y")
        self.assertEqual(len(self.chain.messages), 2)

    def test_empty_chain_summary(self):
        s = self.chain.summary()
        self.assertIn("messages=0", s)
        self.assertIn("agents=set()", s)


class TestRunWriterReviewerLoop(unittest.TestCase):
    @patch.object(llm, "generate_script_with_refinement", return_value="final script")
    def test_returns_script(self, mock_refine):
        result = run_writer_reviewer_loop(
            video_subject="test", language="en", paragraph_number=1
        )
        self.assertEqual(result, "final script")
        mock_refine.assert_called_once()

    @patch.object(llm, "generate_script_with_refinement", return_value="script")
    def test_logs_to_chain_if_provided(self, mock_refine):
        chain = AgentChain(task_id="t1")
        run_writer_reviewer_loop(
            video_subject="test",
            language="en",
            paragraph_number=1,
            chain=chain,
        )
        self.assertGreaterEqual(len(chain.messages), 2)

    @patch.object(llm, "generate_script_with_refinement", return_value="s")
    def test_passes_max_iterations(self, mock_refine):
        run_writer_reviewer_loop(
            video_subject="test",
            language="en",
            paragraph_number=1,
            max_iterations=5,
        )
        mock_refine.assert_called_with(
            video_subject="test",
            language="en",
            paragraph_number=1,
            video_script_prompt="",
            custom_system_prompt="",
            max_refine_iterations=5,
        )


class TestRunKeywordStrategist(unittest.TestCase):
    @patch.object(llm, "generate_terms", return_value=["term1", "term2"])
    def test_returns_terms(self, mock_terms):
        result = run_keyword_strategist(
            video_subject="test", video_script="script", amount=2
        )
        self.assertEqual(result, ["term1", "term2"])
        mock_terms.assert_called_once_with(
            video_subject="test",
            video_script="script",
            amount=2,
            match_script_order=False,
        )

    @patch.object(llm, "generate_terms", return_value=["a", "b"])
    def test_logs_to_chain_if_provided(self, mock_terms):
        chain = AgentChain(task_id="t1")
        run_keyword_strategist(
            video_subject="test",
            video_script="script",
            amount=1,
            chain=chain,
        )
        self.assertEqual(len(chain.messages), 1)

    @patch.object(llm, "generate_terms", return_value=[])
    def test_empty_terms_list(self, mock_terms):
        result = run_keyword_strategist(
            video_subject="test", video_script="script", amount=0
        )
        self.assertEqual(result, [])


class TestRunPipeline(unittest.TestCase):
    @patch.object(llm, "generate_script_with_refinement", return_value="script")
    @patch.object(llm, "generate_terms", return_value=["t1", "t2"])
    def test_returns_script_and_terms(self, mock_terms, mock_refine):
        result = run_pipeline(
            task_id="pipe-1",
            video_subject="test",
            language="en",
            paragraph_number=2,
            amount=2,
        )
        self.assertEqual(result["script"], "script")
        self.assertEqual(result["terms"], ["t1", "t2"])
        self.assertIn("chain_message_count", result)

    @patch.object(llm, "generate_script_with_refinement", return_value="s")
    @patch.object(llm, "generate_terms", return_value=["x"])
    def test_enable_review_passed_to_max_iterations(self, mock_terms, mock_refine):
        run_pipeline(
            task_id="p1",
            video_subject="test",
            language="en",
            paragraph_number=1,
            enable_review=False,
        )
        mock_refine.assert_called_with(
            video_subject="test",
            language="en",
            paragraph_number=1,
            video_script_prompt="",
            custom_system_prompt="",
            max_refine_iterations=0,
        )


if __name__ == "__main__":
    unittest.main()
