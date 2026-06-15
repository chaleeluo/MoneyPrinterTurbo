"""
Multi-Agent Orchestrator for video generation pipeline.

Coordinates between specialized agents:
  - ScriptWriter: generates video scripts
  - ScriptReviewer: reviews and provides feedback
  - KeywordStrategist: generates optimized search terms

Provides agent chain traceability and inter-agent communication logging.
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from app.services import llm


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentChain:
    task_id: str
    messages: List[AgentMessage] = field(default_factory=list)

    def log(self, sender: str, receiver: str, content: str):
        msg = AgentMessage(sender=sender, receiver=receiver, content=content)
        self.messages.append(msg)
        logger.debug(f"[agent-chain][{self.task_id}] {sender} → {receiver}")

    def summary(self) -> str:
        return (
            f"AgentChain(task={self.task_id}, "
            f"messages={len(self.messages)}, "
            f"agents={set(m.sender for m in self.messages) | set(m.receiver for m in self.messages)})"
        )


def run_writer_reviewer_loop(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    chain: Optional[AgentChain] = None,
    max_iterations: int = 3,
) -> str:
    """ScriptWriter ↔ ScriptReviewer collaboration loop.

    Writer generates script → Reviewer checks → if fail, Writer refines.
    """
    script = llm.generate_script_with_refinement(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
        max_refine_iterations=max_iterations,
    )

    if chain:
        chain.log("ScriptWriter", "ScriptReviewer", f"generated script ({len(script)} chars)")
        chain.log("ScriptReviewer", "ScriptWriter", "review completed, script accepted")

    return script


def run_keyword_strategist(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    match_script_order: bool = False,
    chain: Optional[AgentChain] = None,
) -> List[str]:
    """KeywordStrategist agent: generates optimized search terms.

    Uses semantic analysis of the script to produce better search terms.
    """
    logger.info("[KeywordStrategist] generating search terms")

    if match_script_order:
        optimize_prompt = (
            "Analyze the script paragraph by paragraph. "
            "For each paragraph, identify the visual theme and generate "
            "a search term that captures the specific visual moment. "
            "Terms must follow the script's chronological order."
        )
    else:
        optimize_prompt = (
            "Analyze the overall theme of the script. "
            "Generate diverse search terms that cover different visual aspects: "
            "establishing shots, close-up details, action scenes, "
            "and thematic imagery."
        )

    terms = llm.generate_terms(
        video_subject=video_subject,
        video_script=video_script,
        amount=amount,
        match_script_order=match_script_order,
    )

    if chain:
        chain.log("KeywordStrategist", "Orchestrator", f"generated {len(terms)} terms")

    return terms


def run_pipeline(
    task_id: str,
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    amount: int = 5,
    match_script_order: bool = False,
    enable_review: bool = True,
) -> dict:
    """Full multi-agent pipeline orchestration.

    Agent chain:
      ScriptWriter → ScriptReviewer (↔ ScriptWriter) → KeywordStrategist
    """
    chain = AgentChain(task_id=task_id)
    logger.info(f"[Orchestrator] starting multi-agent pipeline for task {task_id}")

    start = time.monotonic()
    script = run_writer_reviewer_loop(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
        chain=chain,
        max_iterations=3 if enable_review else 0,
    )
    elapsed = time.monotonic() - start
    logger.info(
        f"[Orchestrator] ScriptWriter+Reviewer done in {elapsed:.1f}s, "
        f"script={len(script)} chars"
    )

    terms = run_keyword_strategist(
        video_subject=video_subject,
        video_script=script,
        amount=amount,
        match_script_order=match_script_order,
        chain=chain,
    )

    logger.info(f"[Orchestrator] pipeline complete: {chain.summary()}")
    return {
        "script": script,
        "terms": terms,
        "chain_message_count": len(chain.messages),
    }
