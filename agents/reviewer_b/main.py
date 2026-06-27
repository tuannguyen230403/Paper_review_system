import os
import uvicorn
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from executor import ReviewerBExecutor

load_dotenv()

print("\n[ReviewerB Agent] STARTING...")
print("[ReviewerB Agent] ANTHROPIC_API_KEY exists?:", "YES" if os.getenv("ANTHROPIC_API_KEY") else "NO")
print("[ReviewerB Agent] REVIEWER_B_MODEL:", os.getenv("REVIEWER_B_MODEL"))

skill = AgentSkill(
    id="review_independent_critic",
    name="Independent Scientific Critic",
    description="Skeptical reviewer checking rigor, overclaims, and presentation quality",
    tags=["review", "critic", "skeptic", "anthropic"],
)

agent_card = AgentCard(
    name="Reviewer 2 — Claude Independent Critic",
    description="Anthropic Claude agent acting as a skeptical scientific reviewer to prevent blind consensus.",
    url="http://localhost:8002",
    version="2.1.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[skill],
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
)

app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=DefaultRequestHandler(
        agent_executor=ReviewerBExecutor(),
        task_store=InMemoryTaskStore(),
    ),
).build()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)