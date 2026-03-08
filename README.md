# G1 语音助手迁移部署指南

此项目包含在宇树 G1 机器人 (Ubuntu 20 平台) 上运行的本地 AI 语音助手代码。

## 文件结构

- `src/run.py` : 主体运行逻辑代码 (整合了音频 UDP 多播接收、Kokoro TTS 生成以及指令转化 G1 躯体动作控制)。
- `install_g1.sh` : Ubuntu 20 上的部署拉起脚本。
- `rva_reqs.txt` : Windows 端原开发环境的 conda 包备份。

## 先决条件
1. 具备能联网或者已通过 `install_g1.sh` 搭建好基础条件的 G1 上位机环境 (Ubuntu 20)。
2. 确保所有的模型文件（包括大模型、FunASR 和 Kokoro 模型）都在项目根目录的 `models/` 文件夹下方。当前代码 `src/run.py` 严格依赖相对于它所在的 `../models` 目录。

1. 在 G1 上进入 `script` 目录并运行 `install_g1.sh` 脚本以自动装配环境和配置后端模型。
   ```bash
   cd script
   chmod +x install_g1.sh
   ./install_g1.sh
   ```
   *注意：该脚本将会自动解析 `rva_reqs.txt` 进行 pip 安装，下载安装 Ollama 后台服务，并自动读取 `models/Qwen2.5-3B/Modelfile` 注册建立名为 `qwen2.5-3b-local` 的推理模型。*
2. 手动安装 `unitree_sdk2py`: 针对宇树提供的 `unitree_sdk2_python` 库，进到其文件夹并在新环境下运行 `pip install -e .`。

## 运行方案

1. 确保 G1 底层通信开启并且上位机通过 `eth0` 获取到了 `239.168.123.161` 端分发的组播包。
2. 激活环境并启动脚本：
   ```bash
   conda activate rva_g1
   cd src/
   python3 run.py
   ```
3. 对着 G1 的麦克风讲话。G1 确认语音后将通过其内部扬声器作出语音反馈，或者针对特定指令作出行走 (`LocoClient`) 或上肢挥舞动作 (`G1ArmActionClient`)。
