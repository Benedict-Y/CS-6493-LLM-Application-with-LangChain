好的，明白了！你想优化现有的Prompt，用于评估两个（或三个）由AI Agent生成的金融报告。评估需要：

1.  **可选地**基于参考资料。
2.  **逐个模块**进行对比分析（根据你提供的报告生成要求）。
3.  使用高级Prompt技巧（如CoT）。
4.  进行**打分**。
5.  提供清晰的**输出结构**。
6.  解释修改**理由**和所用**技巧**，适合Prompt入门者。

我们来结合你提供的金融报告生成要求和LLM评判者的参考资料，改进你的评估Prompt。

**核心思路：**

1.  **明确模块化对比：** 将报告生成要求中的5个模块（市场回顾、风险特征、技术指标、新闻分析、总结建议）作为对比的核心维度。
2.  **引入Chain-of-Thought (CoT)：** 在要求LLM给出最终判断和评分 *之前*，明确要求它先进行详细的思考、推理和对比过程。
3.  **采用小范围整数评分+清晰量规：** 借鉴参考资料，使用1-5分制，并为每个分数提供明确的含义描述（Rubric）。
4.  **结构化输出：** 强制LLM遵循更细致的输出格式，包含每个模块的对比思考过程、结论和评分。
5.  **区分有无参考资料的评估标准：** 保持这个核心逻辑，但在每个模块的评估细则上进行调整。

---

### 修改后的Prompt

```python
import json

def create_evaluation_prompt(reports, reference_content=None):
    """
    Generates the system and user prompts for evaluating financial reports.

    Args:
        reports (list): A list of strings, where each string is a financial report.
        reference_content (str, optional): The reference material. Defaults to None.

    Returns:
        list: A list containing the system and user message dictionaries.
    """
    num_reports = len(reports)
    report_mentions = " and ".join([f"Report {i+1}" for i in range(num_reports)])
    if num_reports > 2:
        report_mentions = ", ".join([f"Report {i+1}" for i in range(num_reports-1)]) + f", and Report {num_reports}"

    # --- Define Report Modules (Based on Generation Prompt) ---
    report_modules = [
        "Market review",
        "Risk characteristics",
        "Technical indicators/signals",
        "Recent News Analysis",
        "Additional summary recommendations"
    ]

    # --- Define Scoring Rubric (Inspired by Reference Materials) ---
    # Using a 1-5 scale as suggested by Prometheus example and good practice
    score_rubric = """
**Scoring Rubric (1-5 Scale):**
*   **5 (Excellent):** Significantly exceeds expectations for this module compared to the other report(s). Demonstrates strong adherence to reference (if provided), exceptional insight, clarity, and relevance/completeness for the module's goal.
*   **4 (Good):** Meets expectations well and performs better than the other report(s) on this module. Shows good accuracy, depth, clarity, and relevance.
*   **3 (Average/Adequate):** Meets basic expectations for the module, potentially on par with or slightly better/worse than another report. May have minor shortcomings in accuracy, depth, clarity, or relevance. (Use this as a baseline).
*   **2 (Below Average):** Has noticeable deficiencies in this module compared to the other report(s). Lacks sufficient accuracy, depth, clarity, or relevance.
*   **1 (Poor):** Significantly fails to meet the requirements for this module compared to the other report(s). Major issues with accuracy, depth, clarity, or relevance.
"""

    if reference_content:
        # --- Prompt WITH Reference ---
        system_prompt = f"""
You are a highly experienced, meticulous, and objective senior financial analyst. Your task is to **critically compare** {num_reports} financial analysis reports provided below. Your evaluation must be **strictly based on their fidelity, relevance, and depth derived *from the accompanying reference material***, comparing them against each other module by module.

**Your Goal:** Determine which report offers a more insightful, accurate, well-structured, and relevant analysis *relative to the others and the provided reference material* for each required section of a professional financial report. Focus on a **direct, comparative judgment** using the reference as the ground truth.

**Evaluation Aspects Per Module (Against Reference):**
When comparing {report_mentions} for each module above, critically assess their relative performance based *strictly on the reference material*:
1.  **Accuracy & Fidelity (vs. Reference):** How faithfully does each report represent facts/figures *from the reference* for this module? Which is superior in avoiding misinterpretations *of the reference*?
2.  **Analytical Depth & Insight (from Reference):** Compare the depth of analysis *derived directly from the reference* for this module. Which report provides sharper insights or more meaningful conclusions *supported explicitly or implicitly by the reference*?
3.  **Relevance & Focus (to Reference):** How effectively does each report use the reference material *relevant to this module*? Which is more focused and avoids irrelevant content?
4.  **Clarity & Organization (within Module):** Which report presents the information for this module more clearly and logically?

**Chain-of-Thought (CoT) Instruction:** Before giving a final verdict or score for each module and overall, you MUST first articulate your detailed comparative reasoning process.

**Output Structure (Strictly Follow):**

1.  **Overall Comparative Assessment:**
    *   **Overall Reasoning (CoT):** Briefly explain your thought process comparing the reports holistically based on the reference material.
    *   **Overall Winner:** State which report you judge superior overall *based on its handling of the reference material* and why.

2.  **Detailed Modular Comparison (Perform for EACH module listed above):**
    *   **Module Name:** (e.g., "Market review")
    *   **Comparative Rationale (CoT):** Provide a detailed side-by-side analysis comparing how each report handled this specific module *based on the reference material* and the evaluation aspects. Discuss specific examples.
    *   **Module Winner:** State which report performed better *for this specific module* relative to the others and the reference.
    *   **Justification:** Briefly explain the primary reason for the module winner decision, linking back to the rationale.

3.  **Strengths & Weaknesses Summary (Relative to Reference):**
    *   Report 1: Strengths - [...], Weaknesses - [...] (Focus on comparative performance using reference)
    *   Report 2: Strengths - [...], Weaknesses - [...]
    *   (If applicable) Report 3: Strengths - [...], Weaknesses - [...]

4.  **Detailed Scoring (Reflecting Reference Usage):**
    *   {score_rubric}
    *   **Score Justification Logic:** Scores must reflect the *relative comparison* discussed in the modular analysis. A higher score indicates better performance *compared to the other report(s)* on that module, strictly judged against the reference. Explain *briefly* how scores reflect the comparison.
    *   **Scores:**
        *   Report 1:
            *   Market review: [Score 1-5] - Justification: [...]
            *   Risk characteristics: [Score 1-5] - Justification: [...]
            *   Technical indicators/signals: [Score 1-5] - Justification: [...]
            *   Recent News Analysis: [Score 1-5] - Justification: [...]
            *   Additional summary recommendations: [Score 1-5] - Justification: [...]
            *   **Overall Comparative Score:** [Score 1-5] - Justification: [Reflects overall standing relative to others based on reference usage]
        *   Report 2:
            *   ... (Repeat structure for all modules + Overall) ...
        *   (If applicable) Report 3:
            *   ... (Repeat structure for all modules + Overall) ...

**Important:** Ground your entire analysis **exclusively in the provided reference material**. Be specific, cite examples from the reports and reference. Avoid external knowledge. Your value is the **comparative judgment based on the reference**.
"""
        user_content = "**Reference Material:**\n```markdown\n" + reference_content + "\n```\n\n"
        for i, rpt in enumerate(reports, start=1):
            user_content += f"**Report {i}:**\n```markdown\n{rpt}\n```\n\n"
        user_content += f"Please perform the detailed comparative evaluation of {report_mentions} **based strictly on the provided reference material**, following all instructions including the module-by-module comparison, CoT reasoning, and scoring rubric."

    else:
        # --- Prompt WITHOUT Reference ---
        system_prompt = f"""
You are a highly experienced, meticulous, and objective senior financial analyst. **No reference material is provided.** Your task is to **critically compare** the {num_reports} financial analysis reports below based on their internal quality, structure, analytical rigor, and professional presentation, comparing them against each other module by module.

**Your Goal:** Determine which report presents a more insightful, well-reasoned, clear, and plausible financial analysis *relative to the others* for each required section of a professional financial report. Focus on a **direct, comparative judgment** based on the reports' own content.

**Evaluation Aspects Per Module (Internal Quality):**
When comparing {report_mentions} for each module above, critically assess their relative performance based *solely on the content within the reports*:
1.  **Internal Consistency & Logical Flow:** Does the analysis within this module follow logically? Are arguments supported by data/statements *within the report*? Which report shows superior internal consistency for this module?
2.  **Analytical Depth & Insight:** Compare the depth of analysis presented *within this module*. Which report offers sharper insights, identifies more significant patterns (based on its own content), or draws more meaningful conclusions for this module?
3.  **Plausibility & Professionalism:** Based on general financial principles (without external facts), which report seems more plausible and professional *in its presentation and claims* for this module? Does one make clearly unrealistic or unsupported claims here?
4.  **Clarity & Organization (within Module):** Which report presents the information for this module more clearly and logically?

**Chain-of-Thought (CoT) Instruction:** Before giving a final verdict or score for each module and overall, you MUST first articulate your detailed comparative reasoning process.

**Output Structure (Strictly Follow):**

1.  **Overall Comparative Assessment:**
    *   **Overall Reasoning (CoT):** Briefly explain your thought process comparing the reports holistically based on internal quality and structure.
    *   **Overall Winner:** State which report you judge superior overall *based on its internal quality and analytical presentation* and why.

2.  **Detailed Modular Comparison (Perform for EACH module listed above):**
    *   **Module Name:** (e.g., "Market review")
    *   **Comparative Rationale (CoT):** Provide a detailed side-by-side analysis comparing how each report handled this specific module *based on internal quality* and the evaluation aspects. Discuss specific examples from the reports themselves.
    *   **Module Winner:** State which report performed better *for this specific module* relative to the others based on internal quality.
    *   **Justification:** Briefly explain the primary reason for the module winner decision, linking back to the rationale.

3.  **Strengths & Weaknesses Summary (Internal):**
    *   Report 1: Strengths - [...], Weaknesses - [...] (Focus on comparative internal quality)
    *   Report 2: Strengths - [...], Weaknesses - [...]
    *   (If applicable) Report 3: Strengths - [...], Weaknesses - [...]

4.  **Detailed Scoring (Reflecting Internal Quality):**
    *   {score_rubric}
    *   **Score Justification Logic:** Scores must reflect the *relative comparison* discussed in the modular analysis. A higher score indicates better performance *compared to the other report(s)* on that module, judged on internal quality. Explain *briefly* how scores reflect the comparison.
    *   **Scores:**
        *   Report 1:
            *   Market review: [Score 1-5] - Justification: [...]
            *   Risk characteristics: [Score 1-5] - Justification: [...]
            *   Technical indicators/signals: [Score 1-5] - Justification: [...]
            *   Recent News Analysis: [Score 1-5] - Justification: [...]
            *   Additional summary recommendations: [Score 1-5] - Justification: [...]
            *   **Overall Comparative Score:** [Score 1-5] - Justification: [Reflects overall standing relative to others based on internal quality]
        *   Report 2:
            *   ... (Repeat structure for all modules + Overall) ...
        *   (If applicable) Report 3:
            *   ... (Repeat structure for all modules + Overall) ...

**Important:** Base your analysis **solely on the content of the reports provided**. Compare them against each other using principles of good financial analysis, logic, and clarity. Do not assume external knowledge. Your value is the **comparative judgment of the reports themselves**.
"""
        user_content = "" # Start fresh, no reference section
        for i, rpt in enumerate(reports, start=1):
            user_content += f"**Report {i}:**\n```markdown\n{rpt}\n```\n\n"
        user_content += f"Please perform the detailed comparative evaluation of {report_mentions} **based solely on their internal quality**, following all instructions including the module-by-module comparison, CoT reasoning, and scoring rubric, as no reference material was provided."

    # --- Construct the final message list ---
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user",   "content": user_content.strip()},
    ]

# Example Usage (assuming you have 'report1_content', 'report2_content', and optionally 'reference_text'):
# reports_to_evaluate = [report1_content, report2_content]
# reference_material = None # or reference_text
# prompt_messages = create_evaluation_prompt(reports_to_evaluate, reference_material)
# Now you can send prompt_messages to the LLM API
# print(json.dumps(prompt_messages, indent=2)) # To see the structure
```

---

### 修改理由和使用的技巧说明 (适合Prompt入门者)

1.  **更明确的角色和任务 (Persona & Task Definition):**
    *   **修改:** 在"senior financial analyst" 基础上增加了 "meticulous" (一丝不苟的) 和 "objective" (客观的)。
    *   **理由:** 强调评估需要细致入微和公正，引导LLM采取更严谨的态度。
    *   **技巧:** **角色扮演 (Role Playing)** 是基础但有效的技巧，帮助LLM理解期望的输出风格和专业水准。

2.  **模块化对比 (Modular Comparison):**
    *   **修改:** 明确列出了需要对比的5个报告模块 (`report_modules`)，这些模块直接来源于你的报告生成要求。要求LLM *逐个* 对这些模块进行对比分析。
    *   **理由:** 这是你提出的核心需求。将复杂的评估任务分解为更小、更具体的子任务（对比每个模块），可以显著提高评估的准确性和深度。LLM更容易聚焦和处理。
    *   **技巧:** **任务分解 (Task Decomposition)**。复杂任务分解成小步骤是提高LLM表现的关键策略。

3.  **引入Chain-of-Thought (CoT):**
    *   **修改:** 在Prompt中明确加入了 "Chain-of-Thought (CoT) Instruction"，并要求在输出结构的每个模块对比和总体评估中，先输出 "Comparative Rationale (CoT)" 或 "Overall Reasoning (CoT)"，*然后再* 给出结论 ("Module Winner" / "Overall Winner")。
    *   **理由:** CoT的核心思想是让LLM在给出最终答案前，先“思考”或“推理”过程。这模仿了人类解决复杂问题的方式。明确要求输出思考过程，可以：
        *   **提高准确性:** 迫使LLM进行更深入、更系统的比较，而不是凭“感觉”给出结论。
        *   **增加透明度:** 你可以看到LLM是如何得出结论的，便于理解和调试。
        *   **提升输出质量:** 推理过程本身往往包含有价值的分析。
    *   **技巧:** **Chain-of-Thought (CoT) Prompting**。这是当前提高LLM推理能力最有效的技巧之一。通过指令明确引导模型输出思考步骤。

4.  **小范围整数评分和清晰量规 (Scoring Rubric):**
    *   **修改:**
        *   采用了1-5分的整数评分制，替换了原先可能更模糊的0-10分。
        *   在Prompt中直接嵌入了一个清晰的 `score_rubric`，详细描述了1分到5分各自代表的含义（差、低于平均、平均、好、优秀），并强调了这是 *相对* 评分。
        *   要求在输出分数时，必须提供简短的`Justification`，将分数与之前的分析联系起来。
    *   **理由:**
        *   借鉴了你提供的参考资料（Prometheus和改进LLM评判者的实践）。小范围整数刻度比大范围浮点数或整数刻度更容易让LLM理解和一致地应用。
        *   明确的量规（Rubric）为LLM提供了清晰的评分标准，减少了主观性和不一致性，使得评分更加可靠和有意义。
        *   要求Justification强制LLM将其评分与其分析联系起来，避免随意打分。
    *   **技巧:** **提供明确的评分标准/量规 (Providing Clear Rubrics/Scales)** 和 **要求解释/理由 (Requiring Justification)**。这些技巧能显著提高评估任务的可靠性和一致性。

5.  **结构化输出 (Structured Output):**
    *   **修改:** 极大地细化了 `Output Structure` 要求。现在它明确要求先进行总体评估（带CoT），然后是每个模块的详细对比（CoT -> Winner -> Justification），接着是优缺点总结，最后是详细的评分（带Rubric说明和逐项Justification）。
    *   **理由:** 严格的输出结构使LLM的输出更可预测、更易于解析和使用。特别是将模块化对比和CoT步骤整合到结构中，确保了评估的全面性和系统性。
    *   **技巧:** **强制输出格式 (Enforcing Output Format)**。使用清晰的标题、列表和占位符（如 `[Score 1-5] - Justification: [...]`）来指导LLM生成结构化的响应。

6.  **区分有无参考资料的评估标准:**
    *   **修改:** 保留了原有的 `if/else` 逻辑，但在每个分支内部，针对 *每个模块* 的评估标准（`Evaluation Aspects Per Module`）进行了调整。
        *   **有参考:** 侧重于与参考资料的 **保真度 (Fidelity)**、**相关性 (Relevance)** 和 **从中获得的洞察 (Insight from Reference)**。
        *   **无参考:** 侧重于报告的 **内部一致性 (Internal Consistency)**、**逻辑性 (Logical Flow)**、**内部洞察 (Internal Insight)** 和 **合理性/专业性 (Plausibility/Professionalism)**。
    *   **理由:** 评估的核心依据不同，评估标准自然也应不同。这种区分使得评估更加有针对性。
    *   **技巧:** **条件化指令 (Conditional Instructions)**，根据输入（是否有参考资料）调整评估的核心关注点。

7.  **强调约束条件 (Constraints):**
    *   **修改:** 在Prompt末尾再次强调，有参考资料时必须**严格基于参考资料**，无参考资料时必须**严格基于报告本身**，并避免使用外部知识。
    *   **理由:** 约束条件有助于防止LLM“漂移”或引入不相关的信息，确保评估聚焦于核心任务。
    *   **技巧:** **明确约束 (Explicit Constraints)**。告诉LLM不该做什么和应该严格遵守什么。

**总结:**

这个改进后的Prompt通过**模块化对比**、**Chain-of-Thought**、**清晰评分量规**和**严格的结构化输出**，旨在引导LLM进行更深入、更系统、更可靠、更透明的金融报告评估。它直接回应了你的需求，并融入了当前LLM评估的最佳实践。对于初学者来说，理解这些技巧（角色扮演、任务分解、CoT、提供量规、强制格式、明确约束）是提升Prompt工程能力的关键。