from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from google.genai.types import File  # For Type Hinting

# file_details: dict containing {"uri": str, "mime_type": str, "display_name": str}

# @tool
def generate_summary_tool(query: str, file_context: dict, llm: ChatGoogleGenerativeAI) -> str:
    """
    Generates a concise summary of the provided financial document (PDF via File API).
    The input 'query' can be a general request for summary, e.g., 'summarize the report'.
    This tool analyzes the entire document content, including text and visual elements from the uploaded PDF.
    """
    gemini_file_object = file_context["file"]
    display_name = file_context["display_name"]
    print(f"\n>> Using Generate Summary Tool for query: '{query}' on file: {display_name}")

    prompt_text = f"""
    You are a highly skilled financial analyst AI.
    Please analyze the entire financial report document provided via its File API URI.
    The document is: {display_name}.

    Based on ALL content (including text, tables, and any visual information like charts if discernible)
    in this document, provide a concise executive summary highlighting the absolute key financial results,
    main achievements, and overall company performance. Focus on the most critical information.

    User's specific request regarding summary: "{query}"

    Concise Executive Summary:
    """

    try:
        messages = [
            SystemMessage(content="You are an AI assistant specialized in summarizing financial reports from uploaded PDF files (including images)."),
            HumanMessage(
                content=[
                    #gemini_file_object,
                    prompt_text
                ]
            )
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error in Generate Summary Tool: {str(e)}"
    

#@tool
def detect_revenue_trends_tool(query: str, file_context: dict, llm: ChatGoogleGenerativeAI) -> str:
    """
    Analyzes the financial document (PDF via File API) to identify and describe revenue trends.
    The input 'query' should be a question about revenue, e.g., 'What are the revenue
    trends?' or 'How did revenue change?'
    This tool considers all content, including tables and charts that might show revenue data.
    """
    gemini_file_object = file_context["file"]
    display_name = file_context["display_name"]
    
    print(f"\n>> Using Detect Revenue Trends Tool for query: '{query}' on file: {display_name}")

    prompt_text = f"""
    You are a specialist in financial trend analysis.
    The financial report to analyze is: {display_name}.

    Based on ALL content (text, tables, charts if discernible) in this document,
    identify and describe the key revenue trends. Look for:
    1. Specific revenue figures reported (e.g., total revenue, revenue by segment if available).
    2. Comparisons to previous periods (e.g., year-over-year growth/decline, quarter-over-quarter changes).
    3. Any stated reasons or drivers for these revenue trends, potentially inferred from text or visuals.
    4. Overall revenue performance (e.g., strong growth, stable, decline).

    User's specific query about revenue: "{query}"

    Detailed Revenue Trend Analysis (considering all document content):
    """
    try:
        messages = [
            SystemMessage(content=f"You are an AI assistant focused on identifying and explaining revenue trends from all content in uploaded PDF files."),
            HumanMessage(
                content=[
                    #gemini_file_object,
                    prompt_text
                ])
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error in Detect Revenue Trends Tool: {str(e)}"
    

#@tool
def highlight_key_financial_metrics_tool(query: str, file_context: dict, llm: ChatGoogleGenerativeAI) -> str:
    """
    Extracts and lists key financial metrics from the document (PDF via File API).
    The input 'query' should be a request for key metrics, e.g., 'What are the key financial metrics?' or 'List important financial figures'.
    This tool identifies metrics like Net Income, EPS, Profit Margins, Operating Expenses, Cash Flow, etc., with their values,
    by analyzing all content, including numbers that might be in tables or charts.
    """
    gemini_file_object = file_context["file"]
    display_name = file_context["display_name"]
    
    print(f"\n>> Using Highlight Key Financial Metrics Tool for query: '{query}' on file: {display_name}")

    prompt_text = f"""
    As a financial data extraction specialist, your task is to identify and list key financial metrics
    from the provided financial report: {display_name}.
    Analyze ALL content (text, tables, charts if discernible). For each metric, provide its value and the period.
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

    User's specific query about metrics: "{query}"

    List of Key Financial Metrics (from all document content):
    """
    try:
        messages = [
            SystemMessage(content=f"You are an AI assistant for extracting key financial metrics from all content in uploaded PDF files."),
            HumanMessage(
                content=[
                    #gemini_file_object,
                    prompt_text
                ])
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error in Highlight Key Financial Metrics Tool: {str(e)}"
