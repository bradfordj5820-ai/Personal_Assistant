from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

@tool
def internet_search(query: str) -> str:
    """Run an internet search for a given query."""
    return f"Search results for: {query}"

# Define available personas, their titles, and custom system prompts
AVAILABLE_PERSONAS = {
    "Research Assistant": (
        "You are an expert researcher. Your job is to conduct thorough research "
        "and then write a polished report. Use the search tool to gather information."
    ),
    "Code Architect": (
        "You are an expert software architect. Focus on clean code patterns, "
        "system design, and robust debugging guidance."
    ),
    "Financial Analyst": (
        "You are an expert financial analyst. Focus on market data, equity analysis, "
        "and evaluating structured options strategies."
    ),
}

# In-memory checkpointer for multi-user session state
memory_checkpointer = MemorySaver()

def get_agent(persona_name: str):
    """Initializes and returns a deep agent configured for the selected persona."""
    instructions = AVAILABLE_PERSONAS.get(persona_name, AVAILABLE_PERSONAS["Research Assistant"])
    
    return create_deep_agent(
        model="google_genai:gemini-3.7-flash",
        tools=[internet_search],
        system_prompt=instructions,
        checkpointer=memory_checkpointer,
    )

def run_agent_turn(user_id: str, persona_name: str, user_message: str):
    """Executes a single turn for a specific user ID and persona, preserving isolated thread memory."""
    agent = get_agent(persona_name)
    
    config = {
        "configurable": {
            "thread_id": f"user_session_{user_id}_{persona_name}"
        }
    }
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config
    )
    
    # Extract the last message payload
    raw_content = result["messages"][-1].content
    
    # If the model returns structured content blocks (list of dicts), extract the text
    if isinstance(raw_content, list):
        text_parts = []
        for block in raw_content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)
    
    return str(raw_content)
