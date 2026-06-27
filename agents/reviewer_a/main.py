import os
import uvicorn
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from executor import ReviewerAExecutor

load_dotenv()

print("\n[ReviewerA Agent] STARTING...")
print("[ReviewerA Agent] OPENAI_API_KEY exists?:", "YES" if os.getenv("OPENAI_API_KEY") else "NO")
print("[ReviewerA Agent] REVIEWER_A_MODEL:", os.getenv("REVIEWER_A_MODEL"))

skill = AgentSkill(
    id="review_domain_expert",
    name="Domain-specific Reviewer",
    description="Technical domain expert reviewer: evaluates novelty, soundness, and methodology",
    tags=["review", "domain-expert", "openai"],
)

agent_card = AgentCard(
    name="Reviewer 1 — Domain Expert (OpenAI)",
    description="OpenAI GPT-based domain expert. Evaluates technical correctness, novelty, and methodology.",
    url="http://localhost:8001",
    version="2.1.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[skill],
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
)

app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=DefaultRequestHandler(
        agent_executor=ReviewerAExecutor(),
        task_store=InMemoryTaskStore(),
    ),
).build()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)