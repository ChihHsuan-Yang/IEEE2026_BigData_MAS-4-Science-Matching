#AgentVerse/agentverse/environments/base.py
from __future__ import annotations
from agentverse.logging import logger

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel
from typing import Iterable

# from agentverse.agents.agent import Agent

if TYPE_CHECKING:
    from agentverse.agents.base import BaseAgent
    from agentverse.message import Message


class BaseRule(BaseModel):
    pass


class BaseEnvironment(BaseModel):
    """
    Base class for environment.

    Args:
        agents: List of agents
        rule: Rule for the environment
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
        last_messages: Messages from last turn
        rule_params: Variables set by the rule
    """

    agents: List[BaseAgent]
    rule: BaseRule
    max_turns: int = 10
    cnt_turn: int = 0
    last_messages: List[Message] = []
    rule_params: Dict = {}
    result_output_path: Optional[str] = None
    live_output_path: Optional[str] = None
    live_log_sink: Any = None

    @abstractmethod
    async def step(self) -> List[Message]:
        """Run one step of the environment"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the environment"""
        pass

    def report_metrics(self) -> None:
        """Report useful metrics"""
        total_spent = sum([agent.get_spend() for agent in self.agents])
        logger.info(f"Total spent: ${total_spent}")
        pass

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
    def _iter_agent_objects(self):
        """
        Robust iterator over actual agent objects.
        Supports:
          - self.agents as List[BaseAgent]
          - self.agents as Dict[Enum, BaseAgent|List[BaseAgent]]
        """
        agents = getattr(self, "agents", None)
        if agents is None:
            return []

        # tasksolving_env: dict of role -> agent or list[agent]
        if isinstance(agents, dict):
            out = []
            for v in agents.values():
                if v is None:
                    continue
                if isinstance(v, list):
                    out.extend([a for a in v if a is not None])
                else:
                    out.append(v)
            return out

        # simulation_env: list of agents
        if isinstance(agents, list):
            return [a for a in agents if a is not None]

        # fallback
        return []

    def get_spend(self):
        agent_list = self._iter_agent_objects()
        return sum([a.get_spend() for a in agent_list])

    def set_live_log_sink(self, sink: Any) -> None:
        object.__setattr__(self, "live_log_sink", sink)

    def _emit_live_log_event(self, event: Dict[str, Any]) -> None:
        sink = getattr(self, "live_log_sink", None)
        if sink is None:
            return
        try:
            sink(event)
        except Exception as exc:
            logger.warn(f"Live log sink failed: {exc}")

    def _append_log_event(
        self,
        logs: List[Dict[str, Any]],
        event: Dict[str, Any],
    ) -> None:
        logs.append(event)
        self._emit_live_log_event(event)

    def report_metrics(self) -> None:
        agent_list = self._iter_agent_objects()
        #total_spent = sum([a.get_spend() for a in agent_list])

        # keep your existing printing format if you want; simplest:
        #logger.info(f"Total spent: ${total_spent:.6f}")

        """
        for a in agent_list:
            name = getattr(a, "name", a.__class__.__name__)
            logger.info(f"Agent {name}: {a.get_spend_formatted()}")
        """
