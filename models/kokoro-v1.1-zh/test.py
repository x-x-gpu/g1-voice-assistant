# This file is hardcoded to transparently reproduce HEARME_zh.wav
# Therefore it may NOT generalize gracefully to other texts
# Refer to Usage in README.md for more general usage patterns

# pip install kokoro>=0.8.1 "misaki[zh]>=0.8.1"
from kokoro import KModel, KPipeline
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import tqdm
import json
import os

REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'
SAMPLE_RATE = 24000

# Local paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "kokoro-v1_1-zh.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
VOICES_DIR = os.path.join(BASE_DIR, "voices")

# How much silence to insert between paragraphs: 5000 is about 0.2 seconds
N_ZEROS = 5000

# Whether to join sentences in paragraphs 1 and 3
JOIN_SENTENCES = True

VOICE_List = ['zf_001', 'zm_010', 'zf_017', 'zf_059']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# texts = [(
# "Kokoro 是一系列体积虽小但功能强大的 TTS 模型。",
# ), (
# "该模型是经过短期训练的结果，从专业数据集中添加了100名中文使用者。",
# "中文数据由专业数据集公司「龙猫数据」免费且无偿地提供给我们。感谢你们让这个模型成为可能。",
# ), (
# "另外，一些众包合成英语数据也进入了训练组合：",
# "1小时的 Maple，美国女性。",
# "1小时的 Sol，另一位美国女性。",
# "和1小时的 Vale，一位年长的英国女性。",
# ), (
# "由于该模型删除了许多声音，因此它并不是对其前身的严格升级，但它提前发布以收集有关新声音和标记化的反馈。",
# "除了中文数据集和3小时的英语之外，其余数据都留在本次训练中。",
# "目标是推动模型系列的发展，并最终恢复一些被遗留的声音。",
# ), (
# "美国版权局目前的指导表明，合成数据通常不符合版权保护的资格。",
# "由于这些合成数据是众包的，因此模型训练师不受任何服务条款的约束。",
# "该 Apache 许可模式也符合 OpenAI 所宣称的广泛传播 AI 优势的使命。",
# "如果您愿意帮助进一步完成这一使命，请考虑为此贡献许可的音频数据。",
# )]

texts = [(
"你好，我是你的AI助手，很高兴认识你。",
"该 Apache 许可模式也符合 OpenAI 所宣称的广泛传播 AI 优势的使命。",
)]

if JOIN_SENTENCES:
    pass
    # for i in (1, 3):
    #     texts[i] = [''.join(texts[i])]

# Load Config and Model
print(f"Loading model from {MODEL_PATH}")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

model = KModel(config=config, model=MODEL_PATH).to(device).eval()

# English pipeline for phonemes
en_pipeline = KPipeline(lang_code='a', model=model, device=device)

# HACK: Mitigate rushing caused by lack of training data beyond ~100 tokens
# Simple piecewise linear fn that decreases speed as len_ps increases
def speed_callable(len_ps):
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1

#path = Path(__file__).parent
path = Path(r'C:\Users\Xiao\Desktop\Interact_pc\TTS\kokoro-v1.1-zh')

for VOICE_NAME in VOICE_List:
    print(f"Processing voice: {VOICE_NAME}")
    VOICE_PATH = os.path.join(VOICES_DIR, f"{VOICE_NAME}.pt")
    
    def en_callable(text):
        if text == 'Kokoro':
            return 'kˈOkəɹO'
        elif text == 'Sol':
            return 'sˈOl'
        # Use voice path explicitly
        return next(en_pipeline(text, voice=VOICE_PATH, speed=1.0)).phonemes

    # Chinese pipeline
    # Must pass repo_id to trigger v1.1 behavior (Bopomofo+Numbers, en_callable support)
    zh_pipeline = KPipeline(lang_code='z', model=model, device=device, en_callable=en_callable, repo_id=REPO_ID)

    # Manual G2P and Infer loop
    # The pipeline() generator is convenient but hides the internal G2P steps.
    
    wavs = []
    for paragraph in tqdm.tqdm(texts):
        for i, sentence in enumerate(paragraph):
            # 2. G2P
            # pipeline.g2p returns (phonemes, tokens) for v1.1
            ps, _ = zh_pipeline.g2p(sentence)
            if not ps: continue
            
            # 3. Tone Fix
            # v1.1 outputs numbers natively, so no manual mapping needed.
            # But we print for verification.
            if VOICE_NAME == VOICE_List[0]:
                print(f"Phonemes: {ps}")

            # 4. Infer
            # KPipeline.infer(model, ps, pack, speed)
            # pack is stored in zh_pipeline.voice_pack usually, or we load it manually.
            # We already loaded VOICE_PATH logic in previous loop, but we need the pack object.
            # Loading it here to be safe and explicit.
            pack = torch.load(VOICE_PATH, map_location=device, weights_only=True)
            
            audio = KPipeline.infer(model, ps, pack, speed=speed_callable(len(ps)))
            
            if hasattr(audio, 'audio'): audio = audio.audio
            elif hasattr(audio, 'numpy'): audio = audio.numpy()
            if isinstance(audio, torch.Tensor): audio = audio.cpu().numpy()
            if hasattr(audio, 'shape') and audio.ndim == 2 and audio.shape[0] == 1: audio = audio.squeeze(0)

            wav = audio
            
            if i == 0 and wavs and N_ZEROS > 0:
                wav = np.concatenate([np.zeros(N_ZEROS), wav])
            wavs.append(wav)

    sf.write(path / f'HEARME_{VOICE_NAME}.wav', np.concatenate(wavs), SAMPLE_RATE)
    print(f"Saved HEARME_{VOICE_NAME}.wav")