"""
USTC面试官 - 沉浸式实时语音面试平台
基于 Streamlit 的主应用入口
"""

import streamlit as st
import asyncio
import os
from pathlib import Path
from datetime import datetime
from modules.audio_processor import TTS_no_stream, voice_to_text, chunking_tool
from modules.llm_agent import llm_stream_chat
import config

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="USTC面试官",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义样式 ====================
st.markdown("""
<style>
    /* 整体背景 */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 对话框样式 */
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
    }
    
    .assistant-message {
        background-color: #F3E5F5;
        border-left: 5px solid #9C27B0;
    }
    
    /* 按钮样式 */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* 标题样式 */
    h1 {
        color: white;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==================== 初始化 Session State ====================
def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "tts_engine" not in st.session_state:
        st.session_state.tts_engine = TTS_no_stream(api_key=config.STEPFUN_API_KEY)
    
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
    
    if "total_questions" not in st.session_state:
        st.session_state.total_questions = 0
    
    if "audio_enabled" not in st.session_state:
        st.session_state.audio_enabled = True

# ==================== 工具函数 ====================
def add_message(role, content):
    """添加消息到历史记录"""
    st.session_state.messages.append({"role": role, "content": content})
    st.session_state.history.append({"role": role, "content": content})

def display_chat_history():
    """显示聊天历史"""
    for msg in st.session_state.messages:
        css_class = "user-message" if msg["role"] == "user" else "assistant-message"
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{role_icon}</span>
                <strong>{"你" if msg["role"] == "user" else "面试官"}</strong>
            </div>
            <div>{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

def generate_tts_audio(text):
    """生成TTS音频"""
    if not st.session_state.audio_enabled:
        return
    
    try:
        # 创建临时音频目录
        audio_dir = Path("temp_audio")
        audio_dir.mkdir(exist_ok=True)
        
        # 生成唯一的音频文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = audio_dir / f"response_{timestamp}.mp3"
        
        # 生成音频
        if st.session_state.tts_engine.to_speech(text, str(audio_path)):
            # 在 Streamlit 中播放音频
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
            
            # 清理临时文件
            try:
                audio_path.unlink()
            except:
                pass
    except Exception as e:
        st.warning(f"音频生成失败: {e}")

def start_interview():
    """开始面试"""
    st.session_state.interview_started = True
    
    # 初始化面试开场白
    welcome_msg = """你好！我是你的面试官，欢迎参加本次面试。

在开始之前，让我先了解一下你的基本情况。请先做个简短的自我介绍，包括你的姓名、学校、专业以及你应聘的岗位。"""
    
    add_message("assistant", welcome_msg)
    generate_tts_audio(welcome_msg)

# ==================== 主界面 ====================
def main():
    init_session_state()
    
    # 标题
    st.title("🎓 USTC 面试官")
    st.markdown("---")
    
    # ==================== 侧边栏 ====================
    with st.sidebar:
        st.header("⚙️ 控制面板")
        
        # 面试控制
        st.subheader("面试控制")
        if not st.session_state.interview_started:
            if st.button("🚀 开始面试", type="primary"):
                start_interview()
                st.rerun()
        else:
            if st.button("🔄 重新开始", type="secondary"):
                st.session_state.messages = []
                st.session_state.history = []
                st.session_state.interview_started = False
                st.session_state.total_questions = 0
                st.rerun()
        
        st.markdown("---")
        
        # 设置选项
        st.subheader("设置")
        st.session_state.audio_enabled = st.checkbox(
            "🔊 启用语音播放", 
            value=st.session_state.audio_enabled
        )
        
        st.markdown("---")
        
        # 统计信息
        st.subheader("📊 面试统计")
        st.metric("提问次数", st.session_state.total_questions)
        st.metric("对话轮数", len(st.session_state.messages) // 2)
        
        st.markdown("---")
        
        # 帮助信息
        with st.expander("💡 使用说明"):
            st.markdown("""
            **文字输入模式：**
            1. 在输入框中输入你的回答
            2. 点击发送或按 Ctrl+Enter
            
            **语音输入模式：**
            1. 点击"🎤 语音输入"按钮
            2. 按 Enter 开始录音
            3. 说话后再次按 Enter 结束
            
            **功能说明：**
            - 💾 所有对话会自动保存
            - 🔊 可选择是否播放语音
            - 🔄 可随时重新开始面试
            """)
    
    # ==================== 主聊天区域 ====================
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.interview_started:
            # 欢迎界面
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: white;">
                <h2>欢迎来到 USTC 面试官系统</h2>
                <p style="font-size: 1.2rem; margin-top: 1rem;">
                    这是一个基于 AI 的沉浸式面试练习平台<br>
                    点击左侧的"开始面试"按钮开始你的面试之旅
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # 显示聊天历史
            display_chat_history()
    
    # ==================== 输入区域 ====================
    if st.session_state.interview_started:
        st.markdown("---")
        
        # 创建两列布局
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # 文字输入框
            user_input = st.text_area(
                "你的回答：",
                placeholder="在这里输入你的回答，或使用语音输入...",
                height=100,
                key="text_input"
            )
        
        # 按钮行
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
        
        with btn_col1:
            send_btn = st.button("📤 发送", type="primary", use_container_width=True)
        
        with btn_col2:
            voice_btn = st.button("🎤 语音输入", use_container_width=True)
        
        # 处理文字输入
        if send_btn and user_input.strip():
            process_user_input(user_input)
        
        # 处理语音输入
        if voice_btn:
            process_voice_input()

# ==================== 处理函数 ====================
def process_user_input(user_input):
    """处理用户的文字输入"""
    # 添加用户消息
    add_message("user", user_input)
    st.session_state.total_questions += 1
    
    # 创建一个占位符用于流式输出
    with st.spinner("面试官正在思考..."):
        response_placeholder = st.empty()
        full_response = ""
        
        # 流式获取回复
        for partial_response in llm_stream_chat(st.session_state.history[:-1], user_input):
            full_response = partial_response
            # 实时更新显示
            response_placeholder.markdown(f"""
            <div class="chat-message assistant-message">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">🤖</span>
                    <strong>面试官</strong>
                </div>
                <div>{full_response}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 添加完整回复到历史
        # 注意：llm_stream_chat 已经在 history 中添加了 user 消息
        # 我们需要添加 assistant 的回复
        st.session_state.history.append({"role": "assistant", "content": full_response})
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # 生成语音
        generate_tts_audio(full_response)
    
    # 重新运行以刷新界面
    st.rerun()

def process_voice_input():
    """处理语音输入"""
    with st.spinner("准备语音输入..."):
        try:
            # 显示录音指引
            st.info("🎙️ 请按照控制台提示进行录音（按 Enter 开始，再次按 Enter 结束）")
            
            # 创建异步事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 异步调用语音转文字
            text = loop.run_until_complete(
                voice_to_text(config.STEPFUN_API_KEY)
            )
            
            loop.close()
            
            if text:
                st.success(f"识别结果: {text}")
                # 处理识别出的文字
                process_user_input(text)
            else:
                st.error("语音识别失败，请重试")
                
        except Exception as e:
            st.error(f"语音输入出错: {e}")
            st.info("请检查麦克风权限，或使用文字输入")

# ==================== 程序入口 ====================
if __name__ == "__main__":
    main()