import os
import uvicorn
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from executor import AreaChairExecutor

load_dotenv()

print("\n[AreaChairAgent] STARTING...")
print("[AreaChairAgent] OPENAI_API_KEY exists?:", "YES" if os.getenv("OPENAI_API_KEY") else "NO")
print("[AreaChairAgent] JUDGE_MODEL:", os.getenv("JUDGE_MODEL"))
print("[AreaChairAgent] JUDGE_REASONING_EFFORT:", os.getenv("JUDGE_REASONING_EFFORT"))

skill = AgentSkill(
    id="area_chair_meta_review",
    name="Inclusive Area Chair (Meta-Reviewer)",
    description="Blind-reasoning meta-review with bias correction, resolves reviewer disagreements, outputs final JSON decision",
    tags=["area-chair", "meta-review", "judge", "openai", "reasoning"],
)

agent_card = AgentCard(
    name="Area Chair — OpenAI",
    description="Inclusive Area Chair using OpenAI models (gpt-4o / o-series) with bias correction for fair evaluation.",
    url="http://localhost:8003",
    version="2.1.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[skill],
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
)

app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=DefaultRequestHandler(
        agent_executor=AreaChairExecutor(),
        task_store=InMemoryTaskStore(),
    ),
).build()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)