# agentverse/environments/__init__.py
#
# Environment registry for the four released protocols. The upstream AgentVerse
# "simulation" environments (classroom, pokemon, prisoner-dilemma, sde-team) are
# not part of this release and are not imported here.
from agentverse.registry import Registry


env_registry = Registry(name="EnvironmentRegistry")


from .base import BaseEnvironment, BaseRule  # noqa: E402,F401

from .tasksolving_env.basic import BasicEnvironment  # noqa: E402,F401
from .tasksolving_env.single_llm_benchmark import (  # noqa: E402,F401
    SingleLLMBenchmarkEnvironment,
)
from .tasksolving_env.single_agent_reflect import (  # noqa: E402,F401
    SingleAgentReflectEnvironment,
)
from .tasksolving_env.per import PerEnvironment  # noqa: E402,F401
from .tasksolving_env.broadcast_deliberation import (  # noqa: E402,F401
    BroadcastDeliberationEnvironment,
)
