import functools
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool

from src.tools import (
    generate_summary_tool,
    detect_revenue_trends_tool,
    highlight_key_financial_metrics_tool
)

# Basic prompt template for the ReAct agent
REACT_PROMPT_TEMPLATE = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format STRICTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: EXACTLY one of [{tool_names}] (without any square brackets around the tool name itself)
Action Input: the input to the action
Observation: the result to the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""


# Create agent
def create_financial_agent(llm: ChatGoogleGenerativeAI, document_text: str):
    """
    Create a conversational financial agent wiyh custom tools.

    Args:
        llm: The initialized language model.
        document_text: The text content of the financial report.
    """

    # To adapt tool functions for LangChain agent
    # Use functools.partial to "pre-fill" llm and document_text argument.

    # The 'name' and 'description' for the Tool object are taken from
    # decorated function's name and docstring by default if using @tool.
    # If manually creating Tool objects, need specify them.
    # Since @tool have comprehensive docstrings, these will serve as desc.
    summary_tool_with_context = functools.partial(
        generate_summary_tool, llm=llm, document_text=document_text
    )
    revenue_tool_with_context = functools.partial(
        detect_revenue_trends_tool, llm=llm, document_text=document_text
    )
    metrics_tool_with_context = functools.partial(
        highlight_key_financial_metrics_tool, llm=llm, document_text=document_text
    )

    # Create LangChain Tool Objects
    # The agent will see original docstring of the decorated functions as their description
    tools = [
        Tool(
            name="FinancialSummary",    # Name the agent will use
            func=summary_tool_with_context,   # The partially filled function
            description=generate_summary_tool.__doc__  # Explicitly pass docstring
        ),
        Tool(
            name="RevenueTrendAnalysis",
            func=revenue_tool_with_context,
            description=detect_revenue_trends_tool.__doc__
        ),
        Tool(
            name="KeyFinancialMetricsExtraction",
            func=metrics_tool_with_context,
            description=highlight_key_financial_metrics_tool.__doc__
        )
    ]

    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        input_key="input",
        return_messages=True,
        output_key="output"
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors="Check your output and make sure it conforms to the expected ReAct format. Ensure the Action is one of the available tools names.",
        max_iterations=5
        #early_stopping_method="generate"
    )

    print("Financial agent created successfully.")
    return agent_executor
