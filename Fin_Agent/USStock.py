import os  
import yfinance as yf  
from datetime import datetime  
from duckduckgo_search import DDGS  
from langchain_openai import ChatOpenAI  
from langchain.agents import AgentExecutor, create_react_agent  
from langchain.tools import Tool  
from langchain.prompts import PromptTemplate  
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  

# ===== 配置 =====  
OPENAI_API_BASE = "https://api.chatanywhere.tech/v1"  
OPENAI_API_KEY = "sk-bVWnGBsURxbZSojM6pU3Xmogfbiq6q835INsr1OLrHtBWmgy"  
LLM_MODEL = "gpt-3.5-turbo"  # 免费API限制使用特定模型  
DEFAULT_TEMPERATURE = 0.2  
AGENT_MAX_ITERATIONS = 5  # 增加迭代次数，防止过早终止  

# ===== 金融数据工具 =====  
def get_stock_price(ticker: str) -> str:  
    """获取指定股票的当前价格和基本信息"""  
    try:  
        stock = yf.Ticker(ticker)  
        info = stock.info  
        
        # 提取基本信息  
        price = info.get('currentPrice', 'N/A')  
        market_cap = info.get('marketCap', 'N/A')  
        pe_ratio = info.get('trailingPE', 'N/A')  
        
        return f"股票代码: {ticker}\n价格: {price}\n市值: {market_cap}\nP/E比率: {pe_ratio}"  
    except Exception as e:  
        return f"获取{ticker}股票信息时出错: {str(e)}"  

def get_financial_statements(ticker: str) -> str:  
    """获取公司最近的财务报表摘要"""  
    try:  
        stock = yf.Ticker(ticker)  
        
        # 获取收入报表数据而不是弃用的earnings  
        income_stmt = stock.income_stmt  
        if income_stmt is not None and not income_stmt.empty:  
            # 获取最近一年的总收入和净收入  
            total_revenue = income_stmt.loc['Total Revenue'].iloc[-1]  
            net_income = income_stmt.loc['Net Income'].iloc[-1]  
            return f"{ticker}最近年度财务数据:\n营收: {total_revenue}\n净利润: {net_income}"  
        else:  
            return f"未找到{ticker}的财务数据"  
    except Exception as e:  
        return f"获取{ticker}财务报表时出错: {str(e)}"  

# ===== 新闻搜索工具 =====  
def search_company_news(query: str) -> str:  
    """使用DuckDuckGo搜索公司相关新闻"""  
    try:  
        search_query = f"{query} 最新财经新闻 股票"  
        with DDGS() as ddgs:  
            results = list(ddgs.text(search_query, max_results=5))  
        
        if not results:  
            return f"未找到关于{query}的新闻"  
            
        formatted_results = ""  
        for i, result in enumerate(results, 1):  
            title = result.get('title', 'No Title')  
            snippet = result.get('body', 'No Content')  
            url = result.get('href', '#')  
            formatted_results += f"{i}. {title}\n{snippet}\n来源: {url}\n\n"  
            
        return f"关于{query}的最新新闻:\n\n{formatted_results}"  
    except Exception as e:  
        return f"搜索{query}新闻时出错: {str(e)}"  

# ===== 工具集成 =====  
def get_all_tools():  
    """整合所有工具 - 使用英文名称"""  
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

# ===== Agent提示模板 =====  
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

# ===== 构建和运行Agent =====  
def build_finreport_agent(verbose=True):  
    """构建FinReport生成Agent"""  
    # 初始化LLM  
    llm = ChatOpenAI(  
        model_name=LLM_MODEL,  
        openai_api_base=OPENAI_API_BASE,  
        openai_api_key=OPENAI_API_KEY,  
        temperature=DEFAULT_TEMPERATURE,  
        streaming=verbose,  
        callbacks=[StreamingStdOutCallbackHandler()] if verbose else None  
    )  
    
    # 加载所有工具  
    tools = get_all_tools()  
    
    # 获取提示模板  
    prompt = PromptTemplate.from_template(FINREPORT_AGENT_TEMPLATE)  
    
    # 创建Agent  
    agent = create_react_agent(llm, tools, prompt)  
    
    # 创建Agent执行器  
    agent_executor = AgentExecutor(  
        agent=agent,  
        tools=tools,  
        verbose=verbose,  
        handle_parsing_errors=True,  
        max_iterations=AGENT_MAX_ITERATIONS  
    )  
    
    return agent_executor  

def generate_finreport(ticker):  
    """为指定股票生成金融报告"""  
    agent = build_finreport_agent(verbose=True)  
    result = agent.invoke({"input": f"Generate a comprehensive financial report for {ticker}. Include stock analysis, news summary, investment opinion, and risk assessment."})  
    return result['output']  

def save_report(ticker, report_content):  
    """将报告保存到文件"""  
    # 创建保存路径 - 修复路径问题
    save_dir = os.path.join("Fin_Agent", "USStock_Finreport")
    
    # 确保文件夹存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 构建完整文件路径
    filepath = os.path.join(save_dir, f"{ticker}_FinReport.md")
    
    with open(filepath, "w", encoding="utf-8") as f:  
        f.write(report_content)  
    return filepath

# ===== 单独测试工具函数 =====  
def test_financial_tools():  
    """测试金融工具功能"""  
    ticker = "AAPL"  
    print(f"测试获取股票价格 - {ticker}:")  
    print(get_stock_price(ticker))  
    print("\n测试获取财务报表:")  
    print(get_financial_statements(ticker))  

def test_news_tools():  
    """测试新闻搜索工具功能"""  
    query = "特斯拉"  
    print(f"测试搜索公司新闻 - {query}:")  
    print(search_company_news(query))  

# ===== 主程序 =====  
def main():  
    print("===== FinReport生成器 =====")  
    print("这个工具可以为任何公司生成专业的金融分析报告")  
    
    while True:  
        print("\n选择操作：")  
        print("1. 生成公司金融报告")  
        print("2. 测试金融数据工具")  
        print("3. 测试新闻搜索工具")  
        print("0. 退出程序")  
        
        choice = input("请输入选项 (0-3): ")  
        
        if choice == "1":  
            ticker = input("请输入股票代码(如'AAPL'): ")  
            if not ticker:  
                continue  
                
            print(f"\n正在为{ticker}生成金融报告，请稍候...")  
            try:  
                report = generate_finreport(ticker)  
                
                print("\n===== 生成的FinReport =====")  
                print(report)  
                
                # 保存报告  
                filename = save_report(ticker, report)  
                print(f"\n报告已保存到文件: {filename}")  
                
            except Exception as e:  
                print(f"生成报告时发生错误: {str(e)}")  
                
        elif choice == "2":  
            test_financial_tools()  
            
        elif choice == "3":  
            query = input("请输入要搜索新闻的公司名称或股票代码: ")  
            if query:  
                print(search_company_news(query))  
                
        elif choice == "0":  
            break  
            
        else:  
            print("无效的选择，请重试")  
    
    print("感谢使用FinReport生成器!")  

if __name__ == "__main__":  
    main()  