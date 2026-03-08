import os
import logging
import time
import math
import socket
import struct
import numpy as np
import noisereduce as nr
import torch
import soundfile as sf
import json
import re
import ollama
import scipy.signal

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
except ImportError:
    print("Warning: unitree_sdk2py module not found. It is required for running on G1.")
    action_map = {}

# Note: These imports are for the other components (VAD, ASR, TTS) which we are keeping.
from funasr import AutoModel
# Removed modelscope imports for LLM
from kokoro import KModel, KPipeline

# ==========================================
# 日志与环境配置
# ==========================================
logging.basicConfig(level=logging.ERROR)
logging.getLogger("funasr").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("kokoro").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

# ==========================================
# 模型路径配置
# ==========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# FunASR 模型路径
MODEL_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, r'../models'))
os.environ['MODELSCOPE_CACHE'] = MODEL_ROOT_DIR
os.environ['MODELSCOPE_DISABLE_REMOTE'] = '1'

# LLM 配置 (Ollama)
OLLAMA_MODEL_NAME = "qwen2.5-3b-local"

# TTS 模型路径
TTS_BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, r'../models/kokoro-v1.1-zh'))
TTS_MODEL_PATH = os.path.join(TTS_BASE_DIR, "kokoro-v1_1-zh.pth")
TTS_CONFIG_PATH = os.path.join(TTS_BASE_DIR, "config.json")
TTS_VOICE_NAME = 'zf_059'
TTS_VOICE_PATH = os.path.join(TTS_BASE_DIR, "voices", f"{TTS_VOICE_NAME}.pt")
TTS_REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'

# ==========================================
# G1 音频与控制参数配置
# ==========================================
RATE = 16000
TTS_RATE = 24000
CHANNELS = 1

AUDIO_SUBSCRIBE_TOPIC = "rt/audio_msg"
GROUP_IP = "239.168.123.161"
PORT = 5555
NETWORK_INTERFACE = "eth0" 

CHUNK_MS = 200
CHUNK_SIZE = int(RATE * CHUNK_MS / 1000)
SV_THRESHOLD = 0.5

# ==========================================
# Tools Definition
# ==========================================
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "set_velocity",
            "description": "控制机器人底盘移动和旋转。前后移动设置vx，左右平移设置vy，原地旋转/转圈/转弯设置vyaw。",
            "parameters": {
                "type": "object",
                "properties": {
                    "vx": {"type": "number", "description": "X轴速度(m/s)"},
                    "vy": {"type": "number", "description": "Y轴速度(m/s)"},
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

# ==========================================
# 辅助函数
# ==========================================
# arm_action 合法的 command 值
VALID_ARM_COMMANDS = {
    "release arm", "shake hand", "high five", "hug",
    "high wave", "clap", "face wave", "left kiss",
    "heart", "right heart", "hands up", "reject"
}

def validate_tool_call(name, args):
    if name == "arm_action":
        cmd = args.get("command", "") if isinstance(args, dict) else ""
        return cmd in VALID_ARM_COMMANDS
    if name == "set_velocity":
        if isinstance(args, dict):
            return any(k in args for k in ("vx", "vy", "vyaw", "t"))
    return True

def parse_tool_calls_from_text(content):
    tool_calls = []
    known_funcs = ("set_velocity", "arm_action")
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
        else:
            for fn in known_funcs:
                if fn in data and isinstance(data[fn], dict):
                    if validate_tool_call(fn, data[fn]):
                        tool_calls.append({"name": fn, "args": data[fn]})
    return tool_calls

def sanitize_args(args):
    cleaned = {}
    for k, v in args.items():
        if isinstance(v, str):
            try:
                expr = v.replace("Math.PI", str(math.pi)).replace("math.pi", str(math.pi))
                cleaned[k] = eval(expr)
            except:
                cleaned[k] = v
        else:
            cleaned[k] = v
    return cleaned

def speed_callable(len_ps):
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1

def clean_text(text):
    try:
        emoji_pattern = re.compile(r'[\U00010000-\U0010ffff\u2600-\u27bf]', flags=re.UNICODE)
        text = emoji_pattern.sub('', text)
        return text
    except:
        return text

def main():
    print("=" * 60)
    print("正在初始化 AI 语音助手 (G1 SDK Backend)...")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 加载所有模型
    # ------------------------------------------------------------------
    
    print(f"[1/5] Checking Ollama Connection ({OLLAMA_MODEL_NAME})...")
    try:
        ollama.list()
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please ensure 'ollama serve' is running.")
        return

    try:
        print("[2/5] Loading VAD Model...")
        model_vad = AutoModel(
            model=os.path.join(MODEL_ROOT_DIR, "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
            local_files_only=True, 
            device="cuda",
            disable_pbar=True, 
            disable_update=True
        )
    except Exception as e:
        print(f"Error loading VAD: {e}")
        return

    try:
        print("[3/5] Loading ASR Model...")
        model_asr = AutoModel(
            model=os.path.join(MODEL_ROOT_DIR, "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
            local_files_only=True, 
            device="cuda",
            disable_pbar=True, 
            disable_update=True
        )
    except Exception as e:
        print(f"Error loading ASR: {e}")
        return

    try:
        print(f"[4/5] Loading Kokoro TTS ({TTS_VOICE_NAME})...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        with open(TTS_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        model_tts = KModel(config=config, model=TTS_MODEL_PATH).to(device).eval()
        voice_pack = torch.load(TTS_VOICE_PATH, map_location=device, weights_only=True)
        en_pipeline = KPipeline(lang_code='a', model=False) 
        
        def en_callable(text):
            if text == 'Kokoro': return 'kˈOkəɹO'
            elif text == 'Sol': return 'sˈOl'
            return next(en_pipeline(text, voice=TTS_VOICE_PATH, speed=1.0)).phonemes

        zh_pipeline = KPipeline(lang_code='z', model=model_tts, device=device, 
                                repo_id=TTS_REPO_ID, en_callable=en_callable)
        
        zh_pipeline.voices[TTS_VOICE_NAME] = voice_pack
        
    except Exception as e:
        print(f"Error loading TTS: {e}")
        return

    # ------------------------------------------------------------------
    # 2. 初始化 G1控制端 及音频流
    # ------------------------------------------------------------------
    print(f"\n[5/5] 初始化 G1 SDK 控制接口及 Audio Client...")
    try:
        ChannelFactoryInitialize(0, NETWORK_INTERFACE)
        
        loco_client = LocoClient()
        loco_client.Init()
        loco_client.SetTimeout(10.0)
        
        arm_client = G1ArmActionClient()
        arm_client.Init()
        arm_client.SetTimeout(10.0)
        
        audio_client = AudioClient()
        audio_client.Init()
        audio_client.SetTimeout(10.0)
        audio_client.SetVolume(100)
    except Exception as e:
        print(f"Error initializing G1 SDK: {e}")
        # In case user tests on Windows without SDK, we proceed anyway but without action
        loco_client = None
        arm_client = None
        audio_client = None

    # 初始化用于收音的 UDP Socket (获取 G1 组播音频)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        sock.bind(('', PORT))
        mreq = socket.inet_aton(GROUP_IP) + socket.inet_aton('0.0.0.0')
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.setblocking(False)
    except Exception as e:
        print(f"Error configuring UDP socket for G1 mic: {e}")
        return

    print("\n所有模型加载及网络初始化完成！")

    def play_tts(text):
        if not text or not audio_client: return
        text = text.replace('\n', ' ').strip()
        print(f"    [TTS] Generating & Sending to G1: {text[:30]}...")
        try:
            sentences = re.split(r'(?<=。|！|？|\.|!|\?)\s*', text)
            sentences = [s for s in sentences if s.strip()]

            for i, sentence in enumerate(sentences):
                ps, _ = zh_pipeline.g2p(sentence)
                if not ps: continue
                speed = speed_callable(len(ps))

                pack = zh_pipeline.voices[TTS_VOICE_NAME]
                audio = KPipeline.infer(model_tts, ps, pack, speed=speed)

                if hasattr(audio, 'audio'): audio = audio.audio
                elif hasattr(audio, 'numpy'): audio = audio.numpy()
                if isinstance(audio, torch.Tensor): audio = audio.cpu().numpy()
                
                # 重采样 (由 24kHz 到 G1 需要的 16kHz)
                if TTS_RATE != 16000:
                    num_samples = int(len(audio) * 16000 / TTS_RATE)
                    audio = scipy.signal.resample(audio, num_samples)
                
                audio_int16 = (audio * 32767).astype(np.int16)
                if audio_int16.ndim == 2 and audio_int16.shape[0] == 1:
                     audio_int16 = audio_int16.squeeze(0)
                
                if i == 0:
                     silence_length = int(16000 * 0.2)
                     silence = np.zeros(silence_length, dtype=np.int16)
                     audio_int16 = np.concatenate((silence, audio_int16))
                     
                audio_bytes = audio_int16.tobytes()
                
                # 按照数据分块发送给 G1 (防止 SDK 或网络缓冲区溢出)
                chunk_size = 64000 # 约 2 秒音频
                offset = 0
                stream_id = str(int(time.time() * 1000)) + str(i)
                while offset < len(audio_bytes):
                    end = min(offset + chunk_size, len(audio_bytes))
                    chunk = audio_bytes[offset:end]
                    # Convert to bytes array as requested by G1 PlayStream Python SDK
                    audio_client.PlayStream("tts", stream_id, bytes(chunk))
                    offset += chunk_size
                    time.sleep(1)
        except Exception as e:
            print(f"    [TTS Error]: {e}")

    # ------------------------------------------------------------------
    # 3. 实时处理循环
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"【AI 语音助手】已启动 | 音色: {TTS_VOICE_NAME} | 平台: G1")
    print("等待由网络发送的 G1 麦克风音频...")
    print("=" * 60)

    messages_history = [
        {"role": "system", "content": "你是智能机器人助手小智。\n重要规则：\n1. 日常对话（如问候、聊天、提问）必须用文字回复，不要调用工具。\n2. 只有用户明确要求机器人做物理动作时（走、转、挥手、握手等），才调用工具。\n3. '你好'、'谢谢'等是问候语，用文字回复。\n\n工具使用指南：\n移动：set_velocity。'往前走'→vx=0.3；'往后走/退'→vx=-0.3；'往左走'→vy=0.3,vx=0；'往右走'→vy=-0.3,vx=0。默认t=5。\n旋转：set_velocity，必须设vx=0,vy=0。'转圈'→vyaw=0.5,t=6；'转45度'→vyaw=0.785,t=1。注意：'左转/右转'是旋转用vyaw，'左走/右走'是平移用vy，两者不同。\n手臂：arm_action。挥手=high wave，握手=shake hand，举手=hands up，飞吻=left kiss，比心=heart，拥抱=hug，鼓掌=clap，拒绝/摆手=reject。"}
    ]

    cache_vad = {}
    current_speech_buffer = []
    is_speaking = False
    audio_buffer = bytes()
    
    try:
        while True:
            try:
                raw_data, _ = sock.recvfrom(16384)
                audio_buffer += raw_data
            except BlockingIOError:
                # 暂时没有数据，加个短暂休眠避免CPU 100%
                time.sleep(0.01)
                continue
                
            bytes_per_chunk = CHUNK_SIZE * 2
            while len(audio_buffer) >= bytes_per_chunk:
                chunk_bytes = audio_buffer[:bytes_per_chunk]
                audio_buffer = audio_buffer[bytes_per_chunk:]
                
                chunk_data = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                res_vad = model_vad.generate(input=chunk_data, cache=cache_vad, is_final=False, chunk_size=CHUNK_MS, data_type="sound", fs=RATE)

                if not (res_vad and res_vad[0].get("value")):
                    if is_speaking:
                        current_speech_buffer.append(chunk_data)
                    continue

                for seg in res_vad[0]["value"]:
                    start_ms, end_ms = seg[0], seg[1]

                    if start_ms >= 0 and end_ms < 0:
                        is_speaking = True
                        if not current_speech_buffer:
                            current_speech_buffer.append(chunk_data)

                    elif start_ms < 0 and end_ms >= 0:
                        is_speaking = False
                        current_speech_buffer.append(chunk_data)
                        
                        if current_speech_buffer:
                            full_speech = np.concatenate(current_speech_buffer)
                            current_speech_buffer = [] 
                            
                            if len(full_speech) < RATE * 0.2: continue

                            # ASR 识别
                            try:
                                speech_denoised = nr.reduce_noise(y=full_speech, sr=RATE, stationary=True, prop_decrease=0.75)
                            except:
                                speech_denoised = full_speech

                            res_asr = model_asr.generate(input=speech_denoised)
                            text = res_asr[0]["text"] if res_asr else ""
                            
                            if text:
                                print(f"[{time.strftime('%H:%M:%S')}] [User] {text}")
                                
                                # LLM 回复 (Ollama)
                                if len(text) > 1:
                                    try:
                                        messages_history.append({"role": "user", "content": text})
                                        
                                        response = ollama.chat(
                                            model=OLLAMA_MODEL_NAME,
                                            messages=messages_history,
                                            tools=tools_schema,
                                        )
                                        
                                        msg = response['message']
                                        messages_history.append(msg)
                                        
                                        text_to_speak = ""
                                        executed_tools = False
                                        
                                        # 1. 检查结构化工具调用
                                        if msg.get('tool_calls'):
                                            for tool in msg['tool_calls']:
                                                func_name = tool['function']['name']
                                                args = tool['function']['arguments']
                                                args = sanitize_args(args) if isinstance(args, dict) else args
                                                
                                                print(f"    \033[92m[Action] {func_name} -> {args}\033[0m")
                                                executed_tools = True
                                                
                                                if func_name == "arm_action":
                                                    cmd = args.get('command', 'unknown')
                                                    if cmd in VALID_ARM_COMMANDS and arm_client and cmd in action_map:
                                                        arm_client.ExecuteAction(action_map[cmd])
                                                    if not msg.get('content'):
                                                        text_to_speak += f"好的，做个{cmd}。"
                                                elif func_name == "set_velocity":
                                                    vx = float(args.get('vx', 0))
                                                    vy = float(args.get('vy', 0))
                                                    vyaw = float(args.get('vyaw', 0))
                                                    t = float(args.get('t', 1.0))
                                                    if loco_client:
                                                        loco_client.SetVelocity(vx, vy, vyaw, t)
                                                    if not msg.get('content'):
                                                        text_to_speak += "收到移动指令。"

                                        # 2. 兜底：从文本中解析工具调用
                                        if not executed_tools and msg.get('content'):
                                            fallback_calls = parse_tool_calls_from_text(msg['content'])
                                            if fallback_calls:
                                                for call in fallback_calls:
                                                    args = sanitize_args(call['args']) if isinstance(call['args'], dict) else call['args']
                                                    print(f"    \033[92m[Action] {call['name']} -> {args}\033[0m")
                                                    executed_tools = True
                                                    if call['name'] == "arm_action":
                                                        cmd = args.get('command', 'unknown') if isinstance(args, dict) else 'unknown'
                                                        if cmd in VALID_ARM_COMMANDS and arm_client and cmd in action_map:
                                                            arm_client.ExecuteAction(action_map[cmd])
                                                        text_to_speak += f"好的，做个{cmd}。"
                                                    elif call['name'] == "set_velocity":
                                                        vx = float(args.get('vx', 0) if isinstance(args, dict) else 0)
                                                        vy = float(args.get('vy', 0) if isinstance(args, dict) else 0)
                                                        vyaw = float(args.get('vyaw', 0) if isinstance(args, dict) else 0)
                                                        t = float(args.get('t', 1.0) if isinstance(args, dict) else 1.0)
                                                        if loco_client:
                                                            loco_client.SetVelocity(vx, vy, vyaw, t)
                                                        text_to_speak += "收到移动指令。"

                                        # 3. 检查文本回复
                                        content = msg.get('content', '')
                                        if content and not executed_tools:
                                            print(f"    [AI] {content}")
                                            text_to_speak = content
                                        
                                        # 4. TTS 播放
                                        if text_to_speak:
                                            text_to_speak = clean_text(text_to_speak)
                                            if text_to_speak.strip():
                                                play_tts(text_to_speak)
                                            else:
                                                print("    [TTS] Skipped (Empty after cleaning)")
                                        
                                    except Exception as e:
                                        print(f"    [AI Error]: {e}")
                
    except KeyboardInterrupt:
        print("\n停止运行...")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
