import os  
import yfinance as yf  
from datetime import datetime  
from duckduckgo_search import DDGS  
from langchain_openai import ChatOpenAI  
from langchain.agents import AgentExecutor, create_react_agent  
from langchain.tools import Tool  
from langchain.prompts import PromptTemplate  
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  

# ===== CONFIG =====  
OPENAI_API_BASE = "https://api.chatanywhere.tech/v1"  
OPENAI_API_KEY = "YOUR API KEY"  
LLM_MODEL = "deepseek-v3"  # Free API limits to certain models  
DEFAULT_TEMPERATURE = 0.2  
AGENT_MAX_ITERATIONS = 5  # Increase iterations to prevent early termination  

# ===== Finance Data Tool =====  
def get_stock_price(ticker: str) -> str:  
    """Get the current price and basic info of the specified stock"""  
    try:  
        stock = yf.Ticker(ticker)  
        info = stock.info  
        price = info.get('currentPrice', 'N/A')  
        market_cap = info.get('marketCap', 'N/A')  
        pe_ratio = info.get('trailingPE', 'N/A')  
        return f"Ticker: {ticker}\nPrice: {price}\nMarket Cap: {market_cap}\nP/E Ratio: {pe_ratio}"  
    except Exception as e:  
        return f"Error fetching {ticker} stock info: {str(e)}"  

def get_financial_statements(ticker: str) -> str:  
    """Get the latest company financial statement summary"""  
    try:  
        stock = yf.Ticker(ticker)  
        income_stmt = stock.income_stmt  
        if income_stmt is not None and not income_stmt.empty:  
            total_revenue = income_stmt.loc['Total Revenue'].iloc[-1]  
            net_income = income_stmt.loc['Net Income'].iloc[-1]  
            return f"{ticker} Latest Annual Financial Data:\nRevenue: {total_revenue}\nNet Income: {net_income}"  
        else:  
            return f"No financial data found for {ticker}"  
    except Exception as e:  
        return f"Error fetching {ticker} financial statements: {str(e)}"  

# ===== News Search Tool =====  
def search_company_news(query: str) -> str:  
    """Use DuckDuckGo to search for company news"""  
    try:  
        search_query = f"{query} latest financial news stock"  
        with DDGS() as ddgs:  
            results = list(ddgs.text(search_query, max_results=5))  
        if not results:  
            return f"No news found for {query}"  
        formatted_results = ""  
        for i, result in enumerate(results, 1):  
            title = result.get('title', 'No Title')  
            snippet = result.get('body', 'No Content')  
            url = result.get('href', '#')  
            formatted_results += f"{i}. {title}\n{snippet}\nSource: {url}\n\n"  
        return f"Latest news about {query}:\n\n{formatted_results}"  
    except Exception as e:  
        return f"Error searching news for {query}: {str(e)}"  

# ===== Tool Integration =====  
def get_all_tools():  
    """Integrate all tools - use English names"""  
    tools = [  
        Tool(  
            name="get_stock_price",  
            func=get_stock_price,  
            description="Get the current price and basic market information for a stock. Input should be a valid stock ticker symbol like AAPL."  
        ),  
        Tool(  
            name="get_financial_statements",  
            func=get_financial_statements,  
            description="Get the recent financial statement summary for a company. Input should be a valid stock ticker symbol like AAPL."  
        ),  
        Tool(  
            name="search_company_news",  
            func=search_company_news,  
            description="Search for the latest financial news about a company. Input should be a company name or stock ticker like 'Apple' or 'AAPL'."  
        )  
    ]  
    return tools  

# ===== Agent Prompt Template =====  
FINREPORT_AGENT_TEMPLATE = """  
You are a professional financial analyst assistant. Your goal is to generate a concise yet informative financial report ("FinReport") for a given company.  

You have access to the following tools:  
{tools}  

Follow this format for your thought process and actions:  

Question: the input question you must answer (e.g., Generate a FinReport for AAPL)  
Thought: you should always think about what to do next. First, you need to collect enough information.  
Action: the action to take, should be one of [{tool_names}]  
Action Input: the input to the action  
Observation: the result of the action  
... (this Thought/Action/Action Input/Observation process can repeat several times)  
Thought: I now have enough information to generate a professional financial report.  
Final Answer: Generate a comprehensive and structured financial report with the following sections:  

# Financial Report for [COMPANY]  
## Date: [CURRENT DATE]  

## 1. Company Overview and Stock Performance  
[Include actual stock price, market cap, P/E ratio and other relevant metrics]  

## 2. Financial Analysis  
[Include revenue, net income, and other financial metrics if available]  

## 3. Recent News Analysis  
[Analyze the most relevant recent news and their potential impact]  

## 4. Investment Outlook  
[Provide a balanced assessment based on the collected data]  

## 5. Risk Factors  
[Identify potential risks and challenges]  

Use actual data from your tool observations. Be specific with numbers and facts.  

Start!  

Question: {input}  
{agent_scratchpad}  
"""  

# ===== Build and Run Agent =====  
def build_finreport_agent(verbose=True):  
    """Build the FinReport Agent"""  
    llm = ChatOpenAI(  
        model_name=LLM_MODEL,  
        openai_api_base=OPENAI_API_BASE,  
        openai_api_key=OPENAI_API_KEY,  
        temperature=DEFAULT_TEMPERATURE,  
        streaming=verbose,  
        callbacks=[StreamingStdOutCallbackHandler()] if verbose else None  
    )  
    tools = get_all_tools()  
    prompt = PromptTemplate.from_template(FINREPORT_AGENT_TEMPLATE)  
    agent = create_react_agent(llm, tools, prompt)  
    agent_executor = AgentExecutor(  
        agent=agent,  
        tools=tools,  
        verbose=verbose,  
        handle_parsing_errors=True,  
        max_iterations=AGENT_MAX_ITERATIONS  
    )  
    return agent_executor  

def generate_finreport(ticker):  
    """Generate a financial report for the given stock"""  
    agent = build_finreport_agent(verbose=True)  
    result = agent.invoke({"input": f"Generate a comprehensive financial report for {ticker}. Include stock analysis, news summary, investment opinion, and risk assessment."})  
    return result['output']  

def save_report(ticker, report_content):  
    """Save the report to a file"""  
    save_dir = os.path.join("Fin_Agent", "USStock_Finreport")  
    if not os.path.exists(save_dir):  
        os.makedirs(save_dir)  
    filepath = os.path.join(save_dir, f"{ticker}_FinReport.md")  
    with open(filepath, "w", encoding="utf-8") as f:  
        f.write(report_content)  
    return filepath  

# ===== Test Utility Functions =====  
def test_financial_tools():  
    """Test financial tool functions"""  
    ticker = "AAPL"  
    print(f"Testing get_stock_price - {ticker}:")  
    print(get_stock_price(ticker))  
    print("\nTesting get_financial_statements:")  
    print(get_financial_statements(ticker))  

def test_news_tools():  
    """Test news search tool functions"""  
    query = "Tesla"  
    print(f"Testing search_company_news - {query}:")  
    print(search_company_news(query))  

# ===== Main Program =====  
def main():  
    print("===== FinReport Generator =====")  
    print("This tool can generate a professional financial analysis report for any company")  
    while True:  
        print("\nSelect an option:")  
        print("1. Generate Company Financial Report")  
        print("2. Test Financial Data Tool")  
        print("3. Test News Search Tool")  
        print("0. Exit")  
        choice = input("Please enter your choice (0-3): ")  
        if choice == "1":  
            ticker = input("Enter Stock Ticker (e.g. 'AAPL'): ")  
            if not ticker:  
                continue  
            print(f"\nGenerating financial report for {ticker}, please wait...")  
            try:  
                report = generate_finreport(ticker)  
                print("\n===== Generated FinReport =====")  
                print(report)  
                filename = save_report(ticker, report)  
                print(f"\nReport saved to file: {filename}")  
            except Exception as e:  
                print(f"Error generating report: {str(e)}")  
        elif choice == "2":  
            test_financial_tools()  
        elif choice == "3":  
            query = input("Enter company name or ticker to search news: ")  
            if query:  
                print(search_company_news(query))  
        elif choice == "0":  
            break  
        else:  
            print("Invalid choice, please try again")  
    print("Thank you for using the FinReport Generator!")  

if __name__ == "__main__":  
    main()  
