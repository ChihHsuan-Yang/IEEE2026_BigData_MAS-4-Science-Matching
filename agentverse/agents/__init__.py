# agentverse/agents/__init__.py
#
# Agent registry for the four released protocols. Upstream AgentVerse
# "simulation" agents are not part of this release and are not imported here.
from agentverse.registry import Registry

agent_registry = Registry(name="AgentRegistry")


from .base import BaseAgent  # noqa: E402,F401

from agentverse.agents.tasksolving_agent.role_assigner import (  # noqa: E402,F401
    RoleAssignerAgent,
)
from agentverse.agents.tasksolving_agent.critic import CriticAgent  # noqa: E402,F401
from agentverse.agents.tasksolving_agent.deliberator import (  # noqa: E402,F401
    DeliberationAgent,
)
from agentverse.agents.tasksolving_agent.evaluator import (  # noqa: E402,F401
    EvaluatorAgent,
)
from agentverse.agents.tasksolving_agent.solver import SolverAgent  # noqa: E402,F401
from agentverse.agents.tasksolving_agent.manager import ManagerAgent  # noqa: E402,F401
from agentverse.agents.tasksolving_agent.executor import (  # noqa: E402,F401
    ExecutorAgent,
)
from agentverse.agents.repeater import RepeaterAgent  # noqa: E402,F401
