import json
import time
from typing import Dict, List

# 更新导入路径
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.memory import (
    ConversationBufferMemory, 
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
from langchain.chains import ConversationChain

class ConversationStrategyEvaluator:
    def __init__(self, api_base, api_key):
        self.api_base = api_base
        self.api_key = api_key
        self.test_cases = []
        self.results = {}
        
    def add_test_case(self, name: str, conversation: List[str], expected_tasks: List[str]):
        """添加测试用例: 对话流程和期望完成的任务"""
        self.test_cases.append({
            "name": name,
            "conversation": conversation,
            "expected_tasks": expected_tasks
        })
        
    def run_evaluation(self, memory_types: List[str], params: Dict = None):
        """使用不同对话历史策略运行评估"""
        for memory_type in memory_types:
            print(f"测试对话历史策略: {memory_type}")
            self.results[memory_type] = []
            
            for test_case in self.test_cases:
                memory = self._create_memory(memory_type, params)
                llm = ChatOpenAI(
                    model_name="deepseek-r1",
                    openai_api_base=self.api_base,
                    openai_api_key=self.api_key
                )
                chain = ConversationChain(llm=llm, memory=memory, verbose=False)
                
                # 记录测试数据
                test_result = {
                    "test_name": test_case["name"],
                    "memory_type": memory_type,
                    "exchanges": [],
                    "metrics": {}
                }
                
                # 运行对话
                start_time = time.time()
                for user_input in test_case["conversation"]:
                    response = chain.invoke({"input": user_input})["response"]
                    test_result["exchanges"].append({
                        "user": user_input,
                        "assistant": response
                    })
                end_time = time.time()
                
                # 计算指标
                test_result["metrics"]["response_time"] = end_time - start_time
                test_result["metrics"]["task_completion"] = self._evaluate_task_completion(
                    test_result["exchanges"], test_case["expected_tasks"]
                )
                test_result["metrics"]["context_coherence"] = self._evaluate_coherence(
                    test_result["exchanges"]
                )
                
                self.results[memory_type].append(test_result)
                
    def _create_memory(self, memory_type: str, params: Dict = None):
        """创建不同类型的对话历史记忆组件"""
        params = params or {}
        if memory_type == "buffer":
            return ConversationBufferMemory()
        elif memory_type == "window":
            k = params.get("k", 3)
            return ConversationBufferWindowMemory(k=k)
        elif memory_type == "summary":
            return ConversationSummaryMemory(llm=ChatOpenAI(
                model_name="gpt-3.5-turbo", 
                openai_api_base=self.api_base,
                openai_api_key=self.api_key
            ))
        else:
            raise ValueError(f"未知的对话历史策略: {memory_type}")
    
    def _evaluate_task_completion(self, exchanges, expected_tasks):
        """评估任务完成情况"""
        last_response = exchanges[-1]["assistant"]
        completed_tasks = 0
        for task in expected_tasks:
            if task.lower() in last_response.lower():
                completed_tasks += 1
        return completed_tasks / len(expected_tasks) if expected_tasks else 0
    
    def _evaluate_coherence(self, exchanges):
        """评估上下文连贯性"""
        if len(exchanges) < 2:
            return 1.0
        coherence_score = 0.0
        for i in range(1, len(exchanges)):
            prev_exchange = exchanges[i-1]
            curr_exchange = exchanges[i]
            if any(word in curr_exchange["assistant"].lower() for word in prev_exchange["user"].lower().split()):
                coherence_score += 1.0
        return coherence_score / (len(exchanges) - 1)
    
    def save_results(self, filename: str):
        """保存评估结果"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
            
    def print_summary(self):
        """打印评估结果摘要"""
        for memory_type, results in self.results.items():
            print(f"\n==== {memory_type} 对话历史策略评估结果 ====")
            avg_completion = sum(r["metrics"]["task_completion"] for r in results) / len(results)
            avg_coherence = sum(r["metrics"]["context_coherence"] for r in results) / len(results)
            avg_time = sum(r["metrics"]["response_time"] for r in results) / len(results)
            
            print(f"平均任务完成率: {avg_completion:.2f}")
            print(f"平均上下文连贯性: {avg_coherence:.2f}")
            print(f"平均响应时间: {avg_time:.2f}秒")


# 示例使用
def run_evaluation_example():
    evaluator = ConversationStrategyEvaluator(
        api_base="https://api.chatanywhere.tech/v1",
        api_key="sk-bVWnGBsURxbZSojM6pU3Xmogfbiq6q835INsr1OLrHtBWmgy" 
    )
    
    # 添加多轮对话测试用例
    evaluator.add_test_case(
        name="预约安排测试",
        conversation=[
            "你好，我需要安排一个医生预约",
            "我想在下周三下午3点预约",
            "我需要看牙医",
            "我的名字是张三"
        ],
        expected_tasks=["记住预约时间", "记住预约类型", "记住用户姓名"]
    )
    
    # 运行不同对话历史策略的评估（移除了token_buffer避免tiktoken依赖问题）
    evaluator.run_evaluation(
        memory_types=["buffer", "window", "summary"],
        params={"k": 2}
    )
    
    # 打印结果
    evaluator.print_summary()
    evaluator.save_results("conversation_strategy_results.json")

if __name__ == "__main__":
    # 需要安装的依赖:
    # pip install langchain langchain_openai
    run_evaluation_example()