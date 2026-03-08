import ollama
import json
import re
import math

# ================= 配置区域 =================
# 这里的名字必须和你 'ollama list' 里看到的名字一致
# 如果你还没创建，先运行: ollama run qwen2.5:3b
MODEL_NAME = "qwen2.5-3b-local"

print(f"正在连接本地 Ollama ({MODEL_NAME})...")

# ================= 1. 定义 Tools (Schema) =================
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "set_velocity",
            "description": "控制机器人底盘移动和旋转。前后移动设置vx，左右平移设置vy，原地旋转/转圈/转弯设置vyaw。",
            "parameters": {
                "type": "object",
                "properties": {
                    "vx": {"type": "number", "description": "X轴速度(m/s)，前进为正，后退为负"},
                    "vy": {"type": "number", "description": "Y轴速度(m/s)，左移为正，右移为负"},
                    "vyaw": {"type": "number", "description": "旋转角速度(rad/s)，用于转圈/转弯/转向。正值左转，负值右转。0.5慢速转，1.0中速转"},
                    "t": {"type": "number", "description": "持续时间(s)"}
                },
                "required": ["vx", "vy", "vyaw", "t"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "arm_action",
            "description": "执行上肢/手臂动作。中文指令映射：挥手=high wave，握手=shake hand，举手/双手举高=hands up，拥抱=hug，击掌=high five，鼓掌=clap，飞吻=left kiss，比心=heart，右手比心=right heart，拒绝/摆手=reject，放下手臂=release arm",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [
                            "release arm", "shake hand", "high five", "hug", 
                            "high wave", "clap", "face wave", "left kiss", 
                            "heart", "right heart", "hands up", "reject"
                        ]
                    }
                },
                "required": ["command"]
            }
        }
    }
]

# ================= 2. 从文本中解析工具调用（兜底） =================
# arm_action 合法的 command 值
VALID_ARM_COMMANDS = {
    "release arm", "shake hand", "high five", "hug",
    "high wave", "clap", "face wave", "left kiss",
    "heart", "right heart", "hands up", "reject"
}

def validate_tool_call(name, args):
    """验证工具调用参数是否合法"""
    if name == "arm_action":
        cmd = args.get("command", "") if isinstance(args, dict) else ""
        return cmd in VALID_ARM_COMMANDS
    if name == "set_velocity":
        # 至少要有一个速度参数
        if isinstance(args, dict):
            return any(k in args for k in ("vx", "vy", "vyaw", "t"))
    return True

def parse_tool_calls_from_text(content):
    """当模型将工具调用输出为文本时，尝试从中提取调用信息"""
    tool_calls = []
    known_funcs = ("set_velocity", "arm_action")
    
    # 尝试找到所有 JSON 对象
    tag_pattern = r'<tool_(?:call|code)>\s*(\{.*?\})\s*</tool_(?:call|code)>'
    json_block_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    
    all_json_strs = re.findall(tag_pattern, content, re.DOTALL)
    if not all_json_strs:
        all_json_strs = re.findall(json_block_pattern, content, re.DOTALL)
    
    for json_str in all_json_strs:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            continue
        
        # 格式A: {"name": "arm_action" 或 "{arm_action}", "arguments": {...}}
        if "name" in data and "arguments" in data:
            raw_name = data["name"].strip("{} ")
            args = data["arguments"]
            
            if isinstance(args, dict):
                for fn in known_funcs:
                    if fn in args and isinstance(args[fn], dict):
                        if validate_tool_call(fn, args[fn]):
                            tool_calls.append({"name": fn, "args": args[fn]})
                        break
                else:
                    if raw_name in known_funcs and validate_tool_call(raw_name, args):
                        tool_calls.append({"name": raw_name, "args": args})
                    elif "tool" in raw_name:
                        for fn in known_funcs:
                            if fn in args:
                                if validate_tool_call(fn, args[fn]):
                                    tool_calls.append({"name": fn, "args": args[fn]})
        
        # 格式B: {"set_velocity": {...}} 或 {"arm_action": {...}}
        else:
            for fn in known_funcs:
                if fn in data and isinstance(data[fn], dict):
                    if validate_tool_call(fn, data[fn]):
                        tool_calls.append({"name": fn, "args": data[fn]})
    
    return tool_calls

def sanitize_args(args):
    """清理参数中的非法值（如 Math.PI/4 → 数值）"""
    cleaned = {}
    for k, v in args.items():
        if isinstance(v, str):
            # 尝试将字符串表达式转为数值
            try:
                # 替换 Math.PI 为 math.pi
                expr = v.replace("Math.PI", str(math.pi)).replace("math.pi", str(math.pi))
                cleaned[k] = eval(expr)
            except:
                cleaned[k] = v
        else:
            cleaned[k] = v
    return cleaned

# ================= 3. 主对话逻辑 =================
def chat_loop():
    messages = [
        {"role": "system", "content": (
            "你是智能机器人助手小智。\n"
            "重要规则：\n"
            "1. 日常对话（如问候、聊天、提问）必须用文字回复，不要调用工具。\n"
            "2. 只有用户明确要求机器人做物理动作时（走、转、挥手、握手等），才调用工具。\n"
            "3. '你好'、'谢谢'等是问候语，用文字回复。\n\n"
            "工具使用指南：\n"
            "移动：set_velocity。'往前走'→vx=0.3；'往后走/退'→vx=-0.3；'往左走'→vy=0.3,vx=0；'往右走'→vy=-0.3,vx=0。默认t=5。\n"
            "旋转：set_velocity，必须设vx=0,vy=0。'转圈'→vyaw=0.5,t=6；'转45度'→vyaw=0.785,t=1。注意：'左转/右转'是旋转用vyaw，'左走/右走'是平移用vy，两者不同。\n"
            "手臂：arm_action。挥手=high wave，握手=shake hand，举手=hands up，飞吻=left kiss，比心=heart，拥抱=hug，鼓掌=clap，拒绝/摆手=reject。"
        )}
    ]
    
    print(">>> 小智已就绪 (Ollama 版)。直接输入指令，输入 'exit' 退出。")

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        messages.append({"role": "user", "content": user_input})

        try:
            # === 核心调用: 使用 ollama 库 ===
            response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                tools=tools_schema,
            )
            
            msg = response['message']
            executed_tools = False
            
            # 1. 检查是否有结构化工具调用（Ollama 正常解析）
            if msg.get('tool_calls'):
                for tool in msg['tool_calls']:
                    func_name = tool['function']['name']
                    args = tool['function']['arguments']
                    args = sanitize_args(args) if isinstance(args, dict) else args
                    
                    print(f"\033[92m[执行指令] {func_name} -> {args}\033[0m")
                    executed_tools = True

            # 2. 兜底：从文本内容中解析工具调用
            if not executed_tools and msg.get('content'):
                fallback_calls = parse_tool_calls_from_text(msg['content'])
                if fallback_calls:
                    for call in fallback_calls:
                        args = sanitize_args(call['args']) if isinstance(call['args'], dict) else call['args']
                        print(f"\033[92m[执行指令] {call['name']} -> {args}\033[0m")
                        executed_tools = True

            # 3. 显示文本回复（如果没有工具调用，或者有额外文本）
            if msg.get('content') and not executed_tools:
                # 清理掉工具调用相关的文本
                display_text = msg['content']
                display_text = re.sub(r'<tool_(?:call|code)>.*?</tool_(?:call|code)>', '', display_text, flags=re.DOTALL)
                display_text = re.sub(r'\{"name"\s*:\s*"tool_(?:call|code)".*?\}', '', display_text, flags=re.DOTALL)
                display_text = display_text.strip()
                if display_text:
                    print(f"小智: {display_text}")
                
            # 更新历史
            messages.append(msg)

        except Exception as e:
            print(f"发生错误: {e}")
            if "not found" in str(e):
                print(f"请先在终端运行: ollama pull {MODEL_NAME}")

if __name__ == "__main__":
    chat_loop()