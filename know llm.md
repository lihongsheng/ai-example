## LLM 是什么
LLM（大语言模型）本质上基于海量数据进行学习的一个条件概率模型，给它一段输入 token，它预测下一个 token 最可能是什么：
你可以把它当成一个无状态的函数：输入 Prompt，输出文本。每次调用都是独立的，没有记忆，没有状态，对外部世界一无所知。
LLM 就是一个只会不断「猜下一个字」的超级概率计算器， 本身无状态、无记忆、无认知，所有智能都来自海量文本的统计规律。
### LLM 自身的局限性

1. **没有真实理解、认知与语义能力**
   - 本质是统计概率拟合，并非真正“看懂、听懂、思考”。
   - 不了解文字背后对应的真实世界事物、因果关系及物理规则。
   - 仅模仿人类语言规律、句式及知识排列，知其然而不知其所以然。

2. **天生无状态、无记忆、无自我**
   - 模型本身为纯函数（输入 → 输出），无全局变量、无持久化存储。
   - 对话记忆依赖应用层拼接历史上下文实现，非模型原生能力。
   - 缺乏自我意识、主观想法、情感及连续思维。

3. **强上下文依赖，存在窗口上限**
   - 所有需利用的历史内容必须全部填入 Prompt。
   - 上下文窗口有限（如 4k/8k/128k），超长历史会导致截断或遗忘前文。
   - 上下文越长，推理成本、显存占用及耗时呈线性增长。

4. **逻辑薄弱，数学与推理存在短板**
   - 训练数据主要为“语言文本”，而非严谨的数理逻辑。
   - 在简单加减、多步推理及复杂逻辑链中极易出错。
   - 容易跳步或脑补答案，缺乏验算、推导及自我纠错能力。

5. **幻觉问题严重（最致命短板）**
   - 为确保文本流畅、通顺、连贯，可能自动编造内容。
   - 常虚构参考文献、日期、数据、接口、函数或专业结论。
   - 可能一本正经地输出完全错误的事实，且无法自我辨别真假。

6. **无法实时联网，无法感知外部世界**
   - 训练数据为静态离线数据集，存在知识截止时间。
   - 不了解实时新闻、动态数据、当前时间及业务实时状态。
   - 不能主动获取外部信息，仅能依赖“旧数据”。

7. **易受输入引导，顺从性过强**
   - 倾向于顺着用户话术走，无条件迎合提问。
   - 易被诱导生成违规内容、错误结论或恶意文案。
   - Prompt 若稍显模糊或带有误导性，输出结果极易跑偏。

8. **缺乏工具能力，无法直接执行动作**
   - 仅能生成文本，无法直接操作系统、数据库、接口或文件。
   - 不具备计算、查库、调用接口或操作系统的能力。
   - 需额外封装（如 Function Call / 工具调用 / 插件）才能联动外部能力。

9. **输出不可靠、不稳定且随机性强**
   - 相同问题与提示下，多次回答的内容可能不一致。
   - 在敏感、专业或工业场景中，输出结果难以控制。
   - 温度值（temperature）越高，发散性越强，越容易胡编乱造。

10. **无法精准记忆细节，易遗漏关键信息**
    - 对数字、ID、编码、长串字符及精准参数的记忆能力极差。
    - 在长对话中，容易忽略早期的关键约束或前置条件。
    - 面对复杂业务规则或多条件限制时，极易遗漏部分条件。

### 解决LLM自身局性的方式
#### Function Calling
函数调用（Function Calling）是 LLM 处理复杂任务时的核心解决方案，旨在弥补模型自身的局限性：

- **弥补信息滞后**：通过联网工具获取实时信息。
- **增强逻辑计算**：借助代码执行工具解决数学与推理短板。
- **连接外部业务**：利用数据库或接口工具交互真实世界数据。

**核心分工**：
1. **LLM**：仅负责决策调用哪个函数，并提取/填充参数。
2. **代码层**：承担实际的逻辑计算、I/O 操作及接口请求。
```python
from openai import OpenAI
import json
import os

# ======================
# 初始化 OpenAI 客户端
# ======================
client = OpenAI(
    api_key="你的API_KEY",  # 替换成你的 key
    base_url="https://api.openai.com/v1"  # 国内可替换成中转地址
)

# ======================
# 1. 定义本地工具函数
# ======================
def get_current_weather(city: str) -> str:
    """查询天气"""
    weather_data = {
        "北京": "晴天，26℃，微风",
        "上海": "多云，28℃",
        "广州": "小雨，32℃",
        "深圳": "阴天，30℃"
    }
    return json.dumps({
        "city": city,
        "weather": weather_data.get(city, "未知城市")
    }, ensure_ascii=False)


def calculate(a: float, b: float, operator: str) -> str:
    """计算器"""
    if operator == "+":
        res = a + b
    elif operator == "-":
        res = a - b
    elif operator == "*":
        res = a * b
    elif operator == "/":
        res = a / b if b != 0 else "除数不能为0"
    else:
        res = "不支持的运算符"

    return json.dumps({
        "expression": f"{a}{operator}{b}",
        "result": res
    }, ensure_ascii=False)


# ======================
# 2. 函数映射表
# ======================
available_functions = {
    "get_current_weather": get_current_weather,
    "calculate": calculate
}

# ======================
# 3. 定义工具描述（给LLM看）
# ======================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "根据城市名查询当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "进行加减乘除计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "operator": {"type": "string", "enum": ["+", "-", "*", "/"]}
                },
                "required": ["a", "b", "operator"]
            }
        }
    }
]

# ======================
# 4. 主流程：对话 + 自动函数调用
# ======================
def chat_with_function_call(user_query: str):
    # 构造对话历史
    messages = [
        {"role": "user", "content": user_query}
    ]

    # 第一次请求 LLM：判断是否需要调用工具
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # 获取 LLM 回复
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # 如果 LLM 不需要调用工具，直接返回
    if not tool_calls:
        print("✅ LLM直接回答：", response_message.content)
        return

    # ======================
    # 执行 LLM 指定的函数
    # ======================
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(f"🔧 LLM调用函数：{function_name}, 参数：{function_args}")

        # 执行本地函数
        function_response = available_functions[function_name](**function_args)
        print(f"✅ 函数返回结果：{function_response}")

        # 把函数结果加入对话
        messages.append(response_message)  # 加入模型返回
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": function_response
        })

    # ======================
    # 第二次请求 LLM：让模型整理最终答案
    # ======================
    second_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    print("\n🤖 LLM最终回答：", second_response.choices[0].message.content)


# ======================
# 测试
# ======================
if __name__ == "__main__":
    # 测试1：查询天气
    chat_with_function_call("北京今天天气怎么样？")

    # 测试2：计算器
    # chat_with_function_call("3.14 加 5.66 等于多少？")
```

#### AGENT
https://mp.weixin.qq.com/s/eE8PHFRXyTirRMrlJDyZGg?from=singlemessage&isappinstalled=0&scene=1&clicktime=1777298183&enterid=1777298183
https://uaxe.github.io/geektime-docs/AI-%E5%A4%A7%E6%95%B0%E6%8D%AE/AI%E9%87%8D%E5%A1%91%E4%BA%91%E5%8E%9F%E7%94%9F%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91%E5%AE%9E%E6%88%98/02-Agent%E7%9A%84%E5%8E%9F%E7%90%86%EF%BC%9A%E4%BB%80%E4%B9%88%E6%98%AFAI%20Agent%EF%BC%9F/

