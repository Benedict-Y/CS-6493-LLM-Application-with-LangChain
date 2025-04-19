"""
llm_judge.py

用法：
    python llm_judge.py report1.md report2.md [report3.md] --reference reference.md [--model gpt-4] [--temperature 0]

将两个或三个 Markdown 报告文件，连同一个参考信息文件，一并提交给 LLM，
让它以金融领域专家身份进行比较、打分并给出分析结论。
"""

import os
import argparse
import openai
from typing import List

def load_file(path: str) -> str:
    """读取文件内容（支持 Markdown 或纯文本）。"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def construct_messages(reports: List[str], reference: str) -> List[dict]:
    """
    构造 chat completion 的 messages 列表，
    system 提示 LLM 扮演金融领域专家评审，user 提供报告与参考信息。
    """
    system_prompt = (
        "You are a senior financial analyst and evaluator. "
        "Your task is to compare and score financial analysis reports produced by different LLMs. "
        "You should assess each report on criteria such as:\n"
        "  1. Accuracy of financial facts and data interpretation\n"
        "  2. Depth and comprehensiveness of analysis\n"
        "  3. Clarity and organization\n"
        "  4. Relevance to the provided reference information\n"
        "For each report, give a numeric score (0–10) on each criterion, then compute an overall score, "
        "and finally compare the reports with clear reasoning."
    )
    user_content = "参考信息：\n" + reference + "\n\n"
    for i, rpt in enumerate(reports, start=1):
        user_content += f"报告 {i} 内容：\n{rpt}\n\n"
    user_content += (
        "请按照上述标准给出评分，并输出一个对比分析。\n"
        "格式示例：\n"
        "Report 1:\n"
        "- Accuracy: X/10\n"
        "- Depth: X/10\n"
        "- Clarity: X/10\n"
        "- Relevance: X/10\n"
        "- Overall: X/10\n\n"
        "Report 2:\n"
        "... (同上)\n\n"
        "Final Comparison:\n"
        "- 优势/劣势对比\n"
        "- 推荐使用的报告及理由"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

def evaluate_reports(report_paths: List[str], reference_path: str,
                     model: str = "gpt-4", temperature: float = 0.0) -> str:
    """调用 OpenAI API，返回 LLM 的评审结果文本。"""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("请设置环境变量 OPENAI_API_KEY")

    # 读取内容
    reports   = [load_file(p) for p in report_paths]
    reference = load_file(reference_path)

    # 构造对话
    messages = construct_messages(reports, reference)

    # 调用 ChatCompletion
    resp = openai.ChatCompletion.create(
        model=       model,
        messages=    messages,
        temperature= temperature,
        max_tokens=  2000,
    )
    return resp.choices[0].message.content.strip()

def main():
    p = argparse.ArgumentParser(
        description="Use an LLM as judge to compare financial analysis reports."
    )
    p.add_argument(
        "reports",
        nargs="+",
        help="Paths to 2 or 3 markdown report files"
    )
    p.add_argument(
        "--reference", "-r",
        required=True,
        help="Path to reference information file (markdown or text)"
    )
    p.add_argument(
        "--model", "-m",
        default="gpt-4",
        help="OpenAI model to use (e.g. gpt-4, gpt-3.5-turbo)"
    )
    p.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic output)"
    )
    args = p.parse_args()

    if len(args.reports) not in (2, 3):
        p.error("请至少提供 2 个、最多 3 个报告文件路径。")

    try:
        result = evaluate_reports(
            report_paths=    args.reports,
            reference_path=  args.reference,
            model=           args.model,
            temperature=     args.temperature
        )
        print("\n=== LLM 评审结果 ===\n")
        print(result)
    except Exception as e:
        print(f"运行出错：{e}")

if __name__ == "__main__":
    main()
