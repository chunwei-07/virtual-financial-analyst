from langchain.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Full doc text is passed to these tools when initialized/invoked
# Current tools can accept document_text and LLM during exe

#@tool
def generate_summary_tool(query: str, document_text: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Generates a concise summary of the provided financial document
    The input 'query' can be a general request for summary, e.g., 'summarize the report'.
    This tool focuses on overall key takeaways.
    """
    print(f"\n>> Using Generate Summary Tool for query: '{query}'")

    prompt = f"""
    Based on the following financial report text, please provide a concise summary
    highlighting the absolute key financial results, main achievements, and overall
    company performance.
    Focus on the most critical information a busy executive would want to know.
    Avoid excessive detail unless it's crucial for the summary.

    Financial Report Text:
    ---
    {document_text}
    ---

    User's specific request regarding summary: "{query}"

    Concise Executive Summary:
    """

    try:
        messages = [
            SystemMessage(content="You are a highly skilled financial analyst AI assistant specialized in summarizing financial reports."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error in Generate Summary Tool: {str(e)}"
    

#@tool
def detect_revenue_trends_tool(query: str, document_text: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Analyzes the financial document to identify and describe revenue trends.
    The input 'query' should be a question about revenue, e.g., 'What are the revenue
    trends?' or 'How did revenue change?'
    This tool looks for revenue figures, comparisons (e.g., year-over-year, quarter-
    over-quarter), and growth rates.
    """
    print(f"\n>> Using Detect Revenue Trends Tool for query: '{query}'")

    prompt = f"""
    You are a specialist in financial trend analysis. Based on the provided financial report text,
    identify and describe the key revenue trends. Look for:
    1. Specific revenue figures reported (e.g., total revenue, revenue by segment if available).
    2. Comparisons to previous periods (e.g., year-over-year growth/decline, quarter-over-quarter changes).
    3. Any stated reasons or drivers for these revenue trends.
    4. Overall revenue performance (e.g., strong growth, stable, decline).

    Financial Report Text:
    ---
    {document_text}
    ---

    User's specific query about revenue: "{query}"

    Detailed Revenue Trend Analysis:
    """
    try:
        messages = [
            SystemMessage(content=f"You are an AI assistant focused on identifying and explaining revenue trends from financial documents."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error in Detect Revenue Trends Tool: {str(e)}"
    

#@tool
def highlight_key_financial_metrics_tool(query: str, document_text: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Extracts and lists key financial metrics from the document.
    The input 'query' should be a request for key metrics, e.g., 'What are the key financial metrics?' or 'List important financial figures'.
    This tool identifies metrics like Net Income, EPS, Profit Margins, Operating Expenses, Cash Flow, etc., with their values.
    """
    print(f"\n>> Using Highlight Key Financial Metrics Tool for query: '{query}'")

    prompt = f"""
    As a financial data extraction specialist, your task is to identify and list key financial metrics
    from the following financial report text. For each metric, provide its value and the period it pertains to if specified.
    Look for common metrics such as (but not limited to):
    - Total Revenue
    - Net Income / Net Earnings / Profit
    - Earnings Per Share (EPS) - Basic and Diluted
    - Gross Profit & Gross Margin
    - Operating Income & Operating Margin
    - Operating Expenses
    - Cash Flow from Operations
    - Free Cash Flow (FCF)
    - Total Assets & Total Liabilities
    - Shareholders' Equity
    - Key Segment Performance Metrics (if any are prominent)

    Financial Report Text:
    ---
    {document_text}
    ---

    User's specific query about metrics: "{query}"

    List of Key Financial Metrics:
    """
    try:
        messages = [
            SystemMessage(content=f"You are an AI assistant designed to extract and list key financial metrics accurately from reports."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error in Highlight Key Financial Metrics Tool: {str(e)}"