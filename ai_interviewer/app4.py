#这一版整合llm和tts，使用streamlit前端
import streamlit as st
import os
import time
import tempfile
from modules.llm_agent import llm_stream_chat
from modules.audio_processor import TTS_no_stream

# --- 1. 架构配置与初始化 ---
st.set_page_config(page_title="AI 模拟面试官", layout="centered")

# 创建临时音频目录（避免硬编码路径报错）
AUDIO_OUTPUT_DIR = "temp_audio"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# 初始化 Session State (状态管理)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 单例模式初始化 TTS 客户端 (避免每次刷新页面都重新连接 OpenAI)
if "tts_client" not in st.session_state:
    # ⚠️ 架构建议：生产环境中 API Key 应从 st.secrets 或环境变量读取
    # 这里为了演示，沿用你的传参方式
    api_key = "6pZ3jWJGHoMXAcZZpjF3ierYzYDqHEpQLU9gK6auHIWhB1uthsLfqUAnzGLcBiW5x"  
    st.session_state.tts_client = TTS_no_stream(api_key)

import uuid
import re
import time
import os
import shutil

# --- 配置区 ---
MIN_SENTENCE_LEN = 8  # 最小触发字数（包含标点）
AUDIO_OUTPUT_DIR = "temp_audio"

# 启动时自动清理旧音频文件（生产级习惯：保持环境干净）
def clear_temp_audio():
    if os.path.exists(AUDIO_OUTPUT_DIR):
        shutil.rmtree(AUDIO_OUTPUT_DIR)
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

if "initialized" not in st.session_state:
    clear_temp_audio()
    st.session_state.initialized = True

def play_audio(text, index):
    """
    带异常处理的 TTS 调用
    """
    if not text.strip():
        return
    
    # 使用 uuid 确保文件名绝对唯一，防止数十段后出现覆盖错误
    unique_id = uuid.uuid4().hex[:8]
    filename = f"speech_{index}_{unique_id}.mp3"
    filepath = os.path.join(AUDIO_OUTPUT_DIR, filename)
    
    # 调用你的 TTS 模块
    # 注意：建议在你的 TTS 模块里加入小的重试逻辑或 time.sleep(0.1) 避开 API 频率限制
    success = st.session_state.tts_client.to_speech(text, filepath)
    
    if success:
        st.audio(filepath, format="audio/mp3")
    else:
        # 即使失败也静默处理，或者打印日志，不中断用户体验
        st.error(f"TTS 片段 {index} 生成失败，跳过语音")

# --- 核心交互逻辑 ---
if prompt := st.chat_input("请输入你的回答..."):
    # ... (之前的用户输入显示逻辑保持不变)
    # 1. 把用户的输入存入 session_state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # --- 关键修复：定义 chat_history ---
        # 我们把 session_state 里的消息提取出来，过滤掉 Streamlit 特有的 UI 状态
        # 只保留 role 和 content 给 LLM
        chat_history = [
            {"role": m["role"], "content": m["content"]} 
            for m in st.session_state.messages
        ]
        
        # 如果你未来想给 AI 加一个特殊的“系统提示词（System Prompt）”
        # 也可以在这里手动插入到列表的最前面
        # chat_history.insert(0, {"role": "system", "content": "你是一名严厉的面试官"})

        # --- 现在再进行循环就不会报错了 ---
        message_placeholder = st.empty()
        full_response = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 缓冲区逻辑升级
        raw_text_buffer = ""      # 接收 LLM 原始输出的缓冲区
        speech_accumulator = ""   # 专门用于累积“太短的句子”
        last_response_len = 0
        audio_idx = 0

        # 正则表达式：保留标点进行切分
        punc_pattern = r'(?<=[。！？.?!\n])'

        for response in llm_stream_chat(chat_history, prompt):
            delta = response[last_response_len:]
            last_response_len = len(response)
            
            full_response = response
            raw_text_buffer += delta
            
            # 实时更新文字界面
            message_placeholder.markdown(full_response + "▌")

            # 检查是否有完整句子出现
            if any(p in delta for p in "。！？.?!\n"):
                parts = re.split(punc_pattern, raw_text_buffer)
                
                # parts[:-1] 是已经完成的句子，parts[-1] 是还没写完的残句
                for sentence in parts[:-1]:
                    clean_s = sentence.strip()
                    if not clean_s: continue
                    
                    # 将当前句子加入“语音累加器”
                    speech_accumulator += clean_s
                    
                    # --- 核心改进：最小长度判断 ---
                    if len(speech_accumulator) >= MIN_SENTENCE_LEN:
                        # 只有超过长度才触发 TTS
                        play_audio(speech_accumulator, audio_idx)
                        audio_idx += 1
                        speech_accumulator = "" # 清空累加器
                    else:
                        # 长度不够，留在累加器里等下一句
                        print(f"DEBUG: 句子太短 ('{clean_s}')，暂不触发 TTS")
                
                # 剩下的部分重新存入 raw_text_buffer
                raw_text_buffer = parts[-1]

        # --- 收尾工作 (Flush) ---
        # 1. 处理剩下的 raw_text_buffer
        if raw_text_buffer.strip():
            speech_accumulator += raw_text_buffer.strip()
        
        # 2. 只要累加器里还有字，最后必须强行发一次 TTS，否则最后一句会被吞掉
        if speech_accumulator.strip():
            play_audio(speech_accumulator, audio_idx)

        message_placeholder.markdown(full_response)