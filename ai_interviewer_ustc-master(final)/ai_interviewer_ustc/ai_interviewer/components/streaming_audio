"""
流式音频播放组件
使用 HTML5 Audio + JavaScript 实现音频队列播放
"""
import streamlit as st
import base64
import json
from pathlib import Path


def streaming_audio_player(audio_queue: list, key: str = "streaming_audio"):
    """
    流式音频播放器
    
    Args:
        audio_queue: 音频文件路径列表（按顺序）
        key: 组件唯一标识
    """
    if not audio_queue:
        return
    
    # 将音频文件编码为 base64
    audio_data = []
    for audio_path in audio_queue:
        try:
            if Path(audio_path).exists():
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode()
                    audio_data.append({
                        "path": audio_path,
                        "base64": audio_base64
                    })
        except Exception as e:
            print(f"读取音频文件失败 {audio_path}: {e}")
            continue
    
    if not audio_data:
        return
    
    # 将音频数据转为 JSON 字符串
    audio_data_json = json.dumps(audio_data, ensure_ascii=False)
    
    # HTML5 + JavaScript 音频播放器
    html_code = f'''
    <div id="audio-player-{key}" style="margin: 10px 0;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; color: white; margin-bottom: 10px;">
            <strong>🔊 正在播放语音</strong> 
            <span id="audio-status-{key}">(0/{len(audio_data)} 句)</span>
        </div>
        
        <audio id="audio-element-{key}" preload="auto"></audio>
        
        <div style="display: flex; gap: 10px; align-items: center;">
            <button onclick="playAudioQueue_{key}()" 
                    style="background: #10b981; color: white; border: none; 
                           padding: 8px 16px; border-radius: 5px; cursor: pointer;">
                ▶️ 播放
            </button>
            <button onclick="pauseAudio_{key}()" 
                    style="background: #f59e0b; color: white; border: none; 
                           padding: 8px 16px; border-radius: 5px; cursor: pointer;">
                ⏸️ 暂停
            </button>
            <button onclick="stopAudio_{key}()" 
                    style="background: #ef4444; color: white; border: none; 
                           padding: 8px 16px; border-radius: 5px; cursor: pointer;">
                ⏹️ 停止
            </button>
            <div style="flex: 1; background: #e5e7eb; border-radius: 5px; height: 8px; overflow: hidden;">
                <div id="audio-progress-{key}" 
                     style="width: 0%; background: linear-gradient(90deg, #10b981, #059669); 
                            height: 100%; transition: width 0.3s;"></div>
            </div>
        </div>
    </div>
    
    <script>
    (function() {{
        // 音频队列数据
        const audioQueue_{key} = {audio_data_json};

        let currentAudioIndex_{key} = 0;
        let isPlaying_{key} = false;
        let isPaused_{key} = false;
        let hasStarted_{key} = false;  // 标记是否已经开始播放
        
        const audioElement_{key} = document.getElementById('audio-element-{key}');
        const statusElement_{key} = document.getElementById('audio-status-{key}');
        const progressElement_{key} = document.getElementById('audio-progress-{key}');
        
        // 🚀 关键优化：页面加载时自动开始播放（如果有新音频）
        if (!hasStarted_{key}) {{
            hasStarted_{key} = true;
            isPlaying_{key} = true;
            playNext_{key}();
        }}
        
        // 播放下一首
        function playNext_{key}() {{
            if (currentAudioIndex_{key} >= audioQueue_{key}.length) {{
                // 播放完成
                isPlaying_{key} = false;
                statusElement_{key}.textContent = '(完成)';
                progressElement_{key}.style.width = '100%';
                return;
            }}
            
            const audioData = audioQueue_{key}[currentAudioIndex_{key}];
            const audioSrc = 'data:audio/mpeg;base64,' + audioData.base64;
            
            audioElement_{key}.src = audioSrc;
            statusElement_{key}.textContent = '(' + (currentAudioIndex_{key} + 1) + '/' + audioQueue_{key}.length + ' 句)';
            
            // 更新进度条
            const progress = ((currentAudioIndex_{key}) / audioQueue_{key}.length) * 100;
            progressElement_{key}.style.width = progress + '%';
            
            // 播放
            audioElement_{key}.play().then(() => {{
                console.log('正在播放第', currentAudioIndex_{key} + 1, '句');
            }}).catch(error => {{
                console.error('播放失败:', error);
            }});
        }}
        
        // 播放完成事件
        audioElement_{key}.addEventListener('ended', function() {{
            currentAudioIndex_{key}++;
            playNext_{key}();
        }});
        
        // 播放
        window.playAudioQueue_{key} = function() {{
            if (isPlaying_{key} && isPaused_{key}) {{
                // 从暂停恢复
                audioElement_{key}.play();
                isPaused_{key} = false;
            }} else if (!isPlaying_{key}) {{
                // 开始播放
                isPlaying_{key} = true;
                isPaused_{key} = false;
                currentAudioIndex_{key} = 0;
                playNext_{key}();
            }}
        }};
        
        // 暂停
        window.pauseAudio_{key} = function() {{
            if (isPlaying_{key} && !isPaused_{key}) {{
                audioElement_{key}.pause();
                isPaused_{key} = true;
            }}
        }};
        
        // 停止
        window.stopAudio_{key} = function() {{
            audioElement_{key}.pause();
            audioElement_{key}.currentTime = 0;
            isPlaying_{key} = false;
            isPaused_{key} = false;
            currentAudioIndex_{key} = 0;
            statusElement_{key}.textContent = '(已停止)';
            progressElement_{key}.style.width = '0%';
        }};
        
        // 自动开始播放
        setTimeout(() => {{
            console.log('尝试自动播放...');
            window.playAudioQueue_{key}();
        }}, 500);
        
        // 添加手动播放按钮（以防自动播放失败）
        const manualButton = document.createElement('button');
        manualButton.textContent = '🔊 点击播放';
        manualButton.style.cssText = 'padding: 10px 20px; font-size: 16px; cursor: pointer; margin-top: 10px;';
        manualButton.onclick = function() {{
            console.log('手动播放');
            window.playAudioQueue_{key}();
            manualButton.style.display = 'none';
        }};
        container_{key}.appendChild(manualButton);
    }})();
    </script>
    '''
    
    st.components.v1.html(html_code, height=150)


def update_audio_queue_streaming(audio_queue: list, container=None):
    """
    更新音频队列（用于流式更新）
    
    Args:
        audio_queue: 当前的音频路径列表
        container: 可选的 Streamlit 容器
    """
    if container:
        with container:
            streaming_audio_player(audio_queue)
    else:
        streaming_audio_player(audio_queue)
