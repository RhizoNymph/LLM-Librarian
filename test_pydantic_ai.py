from pydantic_ai import Agent
from pydantic import BaseModel

class TestResponse(BaseModel):
    message: str

test_agent = Agent('openai:gpt-4', result_type=TestResponse)

async def test():
    print("Testing PydanticAI...")
    result = await test_agent.run("Say hello!")
    print(f"Result: {result.data.message}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test()) 