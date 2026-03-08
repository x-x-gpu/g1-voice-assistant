#!/bin/bash

# ==============================================================================
# 语音助手 G1 平台 (Ubuntu 20) 环境搭建脚本
# ==============================================================================

set -e

# ================================
# 1. 基础系统依赖
# ================================
echo "[1/5] 安装系统级依赖..."
sudo apt-get update
sudo apt-get install -y build-essential curl wget git bash-completion

# 如果需要支持音频处理及声音接口（即使我们用UDP），建议安装系统基础音频库
sudo apt-get install -y libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# ================================
# 2. 设置 Conda 环境
# ================================
ENV_NAME="rva_g1"
PYTHON_VERSION="3.10"

echo "[2/5] 配置 Conda 环境 ($ENV_NAME)..."
# 检查是否安装了 conda
if ! command -v conda &> /dev/null
then
    echo "未检测到 conda，请先自行安装 Miniconda 或 Anaconda!"
    exit 1
fi

# 初始化 conda bash 支持
source "$(conda info --base)/etc/profile.d/conda.sh"

# 检查环境是否存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 ${ENV_NAME} 已存在，正在激活..."
else
    echo "创建环境 ${ENV_NAME} (Python ${PYTHON_VERSION})..."
    conda create -y -n $ENV_NAME python=${PYTHON_VERSION}
fi

conda activate $ENV_NAME

# ================================
# 3. 安装 Python 核心依赖 (与 Windows rva 环境对齐)
# ================================
echo "[3/5] 安装深度学习及语音核心包..."

# PyTorch (按你的硬件可能需要调整 cuda 版本，这里使用 pip 官方版本)
pip install torch torchvision torchaudio

# 指向项目根目录下的 requirements
pip install -r ../rva_reqs.txt

# ================================
# 4. 安装 Unitree SDK2 Python 版
# ================================
echo "[4/5] 安装宇树 unitree_sdk2py..."
# 如果你已经在机器上有 unitree_sdk2_python 的克隆代码，我们建议在本地安装
# 假设它放在 ~/unitree_sdk2_python，你可以手动 pip install -e ~/unitree_sdk2_python
# 这里提供一个官方 pip 途径或者占位提示
echo "--------------------------------------------------------"
echo "请确保已安装宇树 SDK2 的 Python 环境！"
echo "通常的做法是进入 unitree_sdk2_python 目录，然后执行:"
echo "pip install -e ."
echo "--------------------------------------------------------"

# ================================
# 5. 安装与配置 Ollama (LLM后端)
# ================================
echo "[5/5] 安装与配置 Ollama..."
if ! command -v ollama &> /dev/null
then
    echo "正在安装 Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama 已安装。"
fi

echo "尝试启动 Ollama 后台服务并装载模型..."
nohup ollama serve > ollama_log.txt 2>&1 &
sleep 5  # 等待服务启动

# 使用提供的 Modelfile 创建本地大模型
if [ -f "../models/Qwen2.5-3B/Modelfile" ]; then
    echo "正在基于本地模型注册 qwen2.5-3b-local ..."
    cd ../models/Qwen2.5-3B/
    ollama create qwen2.5-3b-local -f Modelfile
    cd ../../script/
else
    echo "警告：未找到 ../models/Qwen2.5-3B/Modelfile，请检查模型文件是否缺失！"
fi

echo "========================================================="
echo "全部安装完成！"
echo "请使用以下命令激活环境并运行："
echo "conda activate $ENV_NAME"
echo "cd ../src && python3 run.py"
echo "========================================================="
