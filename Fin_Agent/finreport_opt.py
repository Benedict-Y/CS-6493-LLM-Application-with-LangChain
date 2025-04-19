import os  
import re  
import shutil  
import datetime  
from pathlib import Path  
from langchain.chat_models import ChatOpenAI  
from langchain.schema import HumanMessage, SystemMessage  

def setup_directories():  
    """Setup input and output directories."""  
    input_dir = Path("Fin_Agent/Raw Report")  
    output_dir = Path("Fin_Agent/Opt Report")  
    
    # Create output directory if it doesn't exist  
    output_dir.mkdir(parents=True, exist_ok=True)  
    
    return input_dir, output_dir  

def get_md_files(directory):  
    """Get all markdown files in the directory."""  
    return list(directory.glob("*.md"))  

def setup_llm():  
    """Setup LLM client."""  
    return ChatOpenAI(  
        model_name="gpt-4o", 
        openai_api_base="https://api.chatanywhere.tech/v1",  
        openai_api_key="sk-bVWnGBsURxbZSojM6pU3Xmogfbiq6q835INsr1OLrHtBWmgy"  
    )  

def get_user_input():  
    """Get additional information from user."""  
    print("\nWould you like to provide any news or additional information that might impact the stock? (yes/no)")  
    response = input().strip().lower()  
    
    if response == "yes":  
        print("\nPlease provide the news or information:")  
        additional_info = input().strip()  
        return additional_info  
    
    return None  

def optimize_report(file_path, llm, additional_info=None):  
    """Optimize the report using LLM."""  
    # Read the markdown content  
    with open(file_path, 'r', encoding='utf-8') as f:  
        content = f.read()  
    
    # Current date for reference  
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")  
    
    # Prepare system message  
    system_prompt = f"""  
    You are a financial analyst assistant tasked with optimizing financial reports.  
    Current date: {current_date}  
    
    Please analyze the following financial report and:  
    1. Check for any time-related hallucinations (dates, time periods) and correct them based on the current date.  
    2. Verify if there are any discrepancies between image descriptions and actual image content (if you can detect any).  
    3. Maintain the original markdown format including all image links.  
    4. Keep all analytical insights from the original report unless they contain hallucinations.  
    """  
    
    if additional_info:  
        system_prompt += f"""  
        5. Add a new section titled "Recent News Impact Analysis" that analyzes how the following information might impact the stock:  
        
        {additional_info}  
        
        Provide a balanced view of potential positive and negative impacts, considering market reaction.  
        """  
    
    # Messages for the LLM  
    messages = [  
        SystemMessage(content=system_prompt),  
        HumanMessage(content=f"Here is the financial report to optimize:\n\n{content}")  
    ]  
    
    # Get optimized content from LLM  
    response = llm.invoke(messages)  
    
    return response.content  

def save_optimized_report(content, original_path, output_dir):  
    """Save the optimized report."""  
    output_path = output_dir / original_path.name  
    
    with open(output_path, 'w', encoding='utf-8') as f:  
        f.write(content)  
    
    # Check if there are any image directories to copy  
    original_dir = original_path.parent  
    for item in original_dir.iterdir():  
        if item.is_dir() and any(item.glob('*.png')) or any(item.glob('*.jpg')) or any(item.glob('*.jpeg')):  
            # This directory might contain images referenced in the markdown  
            dest_dir = output_dir / item.name  
            if not dest_dir.exists():  
                shutil.copytree(item, dest_dir)  
    
    return output_path  

def main():  
    """Main function to process all reports."""  
    print("Starting financial report optimization process...")  
    
    # Setup  
    input_dir, output_dir = setup_directories()  
    md_files = get_md_files(input_dir)  
    llm = setup_llm()  
    
    if not md_files:  
        print("No markdown files found in the input directory.")  
        return  
    
    print(f"Found {len(md_files)} markdown files to process.")  
    
    # Get additional information from user  
    additional_info = get_user_input()  
    
    # Process each file  
    for i, file_path in enumerate(md_files, 1):  
        print(f"\nProcessing file {i}/{len(md_files)}: {file_path.name}")  
        
        try:  
            # Optimize the report  
            optimized_content = optimize_report(file_path, llm, additional_info)  
            
            # Save the optimized report  
            output_path = save_optimized_report(optimized_content, file_path, output_dir)  
            
            print(f"Successfully optimized and saved to: {output_path}")  
            
        except Exception as e:  
            print(f"Error processing {file_path.name}: {str(e)}")  
    
    print("\nFinancial report optimization completed!")  

if __name__ == "__main__":  
    main()  