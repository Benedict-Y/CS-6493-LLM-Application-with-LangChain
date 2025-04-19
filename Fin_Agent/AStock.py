import os  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from datetime import datetime  

# Add Chinese font support  
plt.rcParams['font.sans-serif'] = ['SimHei']  # For displaying Chinese labels normally  
plt.rcParams['axes.unicode_minus'] = False    # For displaying negative signs normally  

os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"  
os.environ["OPENAI_API_KEY"] = "sk-bVWnGBsURxbZSojM6pU3Xmogfbiq6q835INsr1OLrHtBWmgy"  
LLM_MODEL = "gpt-3.5-turbo"  
DEFAULT_TEMPERATURE = 0.2  

# === I. Data Acquisition (Using Tushare API, simulating real-time data fetching) ========================  
import tushare as ts  

# Your tushare token (replace with your actual token)  
ts.set_token(os.environ.get("TS_TOKEN", "61bc3e2363dbb5362e7181aa355e60f9bf340a0456d2e033db120a8b"))  
pro = ts.pro_api()  

def fetch_stock_data(stock_code='600036.SH', start='20230101', end=None, save_dir='Fin_Agent/Finreport'):  
    if end is None:  
        end = datetime.now().strftime('%Y%m%d')  
    df = pro.daily(ts_code=stock_code, start_date=start, end_date=end)  
    df['trade_date'] = pd.to_datetime(df['trade_date'])  
    df = df.sort_values('trade_date').reset_index(drop=True)  
    out_path = os.path.join(save_dir, f"{stock_code.replace('.', '')}_price.csv")  
    df.to_csv(out_path, index=False,encoding='utf-8-sig')  
    return df, out_path  

# === II. Quantitative Market Analysis and Charts ========================================================  
def get_tech_stats(df):  
    df = df.copy()  
    df['MA5'] = df['close'].rolling(5).mean()  
    df['MA20'] = df['close'].rolling(20).mean()  
    df['daily_return'] = df['close'].pct_change()  
    # Price change percentages  
    def ret(days):   # rolling return utility  
        sdate = df['trade_date'].max() - pd.Timedelta(days=days)  
        sample = df[df['trade_date'] >= sdate]  
        if len(sample) > 1:  
            return sample['close'].iloc[-1] / sample['close'].iloc[0] - 1  
        return np.nan  
    chg_1m = ret(30)  
    chg_3m = ret(90)  
    chg_1y = ret(365)  
    last_close = df['close'].iloc[-1]  
    volatility_1y = df[df['trade_date'] >= df['trade_date'].max() - pd.Timedelta(days=365)]['daily_return'].std() * np.sqrt(252)  
    # Maximum drawdown  
    def max_drawdown(series):  
        roll_max = series.cummax()  
        daily_drawdown = series / roll_max - 1.0  
        return daily_drawdown.min()  
    max_dd = max_drawdown(df['close'])  
    # Moving average golden cross/death cross  
    signal = ''  
    if len(df) > 20:  
        if df['MA5'].iloc[-1] > df['MA20'].iloc[-1] and df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]:  
            signal = 'Short-term moving average golden cross detected, potentially indicating strengthening upward momentum.'  
        elif df['MA5'].iloc[-1] < df['MA20'].iloc[-1] and df['MA5'].iloc[-2] >= df['MA20'].iloc[-2]:  
            signal = 'Short-term moving average death cross detected, caution for pullback risk.'  
        else:  
            signal = 'No significant moving average crossover signals observed.'  
    stats = dict(  
        chg_1m=chg_1m, chg_3m=chg_3m, chg_1y=chg_1y, last_close=last_close,  
        volatility_1y=volatility_1y, max_dd=max_dd, ma_signal=signal  
    )  
    return stats, df  # Return statistics and processed DataFrame  

def save_charts(df, save_dir, prefix='cmb'):  
    # 1. Price + Moving Averages  
    plt.figure(figsize=(10,6))  
    plt.plot(df['trade_date'], df['close'], label='Closing Price')  
    plt.plot(df['trade_date'], df['MA5'], label='MA5', linestyle='--')  
    plt.plot(df['trade_date'], df['MA20'], label='MA20', linestyle='-.')  
    plt.title("Closing Price and Moving Averages")  
    plt.xlabel("Date")  
    plt.ylabel("Price")  
    plt.legend()  
    plt.tight_layout()  
    price_ma_path = os.path.join(save_dir, f"{prefix}_price_ma.png")  
    plt.savefig(price_ma_path, dpi=100)  
    plt.close()  
    # 2. Daily Returns Distribution  
    plt.figure(figsize=(8,5))  
    df['daily_return'].hist(bins=40)  
    plt.xlabel('Daily Return')  
    plt.title('Daily Return Distribution')  
    plt.tight_layout()  
    ret_hist_path = os.path.join(save_dir, f"{prefix}_ret_hist.png")  
    plt.savefig(ret_hist_path)  
    plt.close()  
    return price_ma_path, ret_hist_path  

# === III. LangChain Structured Report Generation ================================================  
from langchain_community.chat_models import ChatOpenAI  
from langchain_core.prompts import ChatPromptTemplate  


def generate_report(stats, stock_code, images, save_path):  
    # Construct Prompt: Inject all structured data, guide LLM to generate professional FinReport  
    prompt_template = """  
You are a senior quantitative financial analyst. Please write a structured professional report based on the following data, including:  
1. Market review (speak with quantitative data, don't fabricate data that's not provided)  
2. Risk characteristics (such as volatility, maximum drawdown)  
3. Technical indicators/signals (such as moving average crossovers, and their potential market implications)  
4. Additional summary recommendations  

【Structured Input Data】  
- Stock Code: {stock_code}  
- 1-year price change: {chg_1y:.2%}  
- 3-month price change: {chg_3m:.2%}  
- 1-month price change: {chg_1m:.2%}  
- Latest closing price: {last_close:.2f} yuan  
- 1-year annualized daily return volatility: {volatility_1y:.2%}  
- Historical maximum drawdown: {max_dd:.2%}  
- Moving average signal analysis: {ma_signal}  

【Available Charts】  
- Price trend and moving averages: {img_price}  
- Daily return distribution: {img_ret}  

Format requirements: Use Markdown syntax, including headings, lists, bold text, etc. to enhance readability. Be concise, quantitative, specific, and maintain a professional style.  
Include the available images in the report, ensuring they are clear.  
"""  
    prompt = prompt_template.format(  
        stock_code = stock_code,  
        chg_1y = stats["chg_1y"],  
        chg_3m = stats["chg_3m"],  
        chg_1m = stats["chg_1m"],  
        last_close = stats["last_close"],  
        volatility_1y = stats["volatility_1y"],  
        max_dd = stats["max_dd"],  
        ma_signal = stats["ma_signal"],  
        img_price = os.path.basename(images[0]),  
        img_ret = os.path.basename(images[1])  
    )  
    chat = ChatOpenAI(  
        temperature=DEFAULT_TEMPERATURE,   
        model=LLM_MODEL,  
        openai_api_key=os.environ["OPENAI_API_KEY"],  
        openai_api_base=os.environ["OPENAI_API_BASE"]  
    )  
    with open(save_path, "w", encoding='utf-8') as f:  
        for r in chat.stream(prompt):  
            print(r.content, end='', flush=True)  # Use .content attribute instead of ['content']  
            f.write(r.content)  

# === IV. Main Process (Ensuring repeatability & directory structure) ================================  
def main():  
    BASE_DIR = r'Fin_Agent\AStock_Finreport'  
    os.makedirs(BASE_DIR, exist_ok=True)  
    stock_code = '600519.SH'  

     # Create subfolder named with stock code + timestamp  
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  
    folder_name = f"{stock_code.replace('.', '')}_{current_time}"  
    SAVE_DIR = os.path.join(BASE_DIR, folder_name)  
    os.makedirs(SAVE_DIR, exist_ok=True)  
    # 1. Fetch data  
    print("Downloading historical data...")  
    df, price_file = fetch_stock_data(stock_code=stock_code, save_dir=SAVE_DIR)  
    print("Historical data saved: ", price_file)  
    # 2. Analysis  
    print("Generating analysis...")  
    stats, df_processed = get_tech_stats(df)  # Receive processed DataFrame  
    # 3. Generate charts  
    price_ma_img, ret_hist_img = save_charts(df_processed, SAVE_DIR, prefix=stock_code.replace('.', ''))  
    # 4. LangChain structured report  
    report_path = os.path.join(SAVE_DIR, f'report_{datetime.now().strftime("%Y%m%d")}_{stock_code}.md')  
    print("Generating LLM report...")  
    generate_report(stats, stock_code, (price_ma_img, ret_hist_img), report_path)  
    print(f"\nReport and charts generation completed: {report_path}")  


if __name__ == "__main__":  
    main()  