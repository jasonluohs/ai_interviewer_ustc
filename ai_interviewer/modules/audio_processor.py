# modules/audio_processor.py
from openai import OpenAI
import os

"""class AudioProcessor:
    def __init__(self, api_key, base_url="https://api.stepfun.com/v1"):
        self.api_key='6pZ3jWJGHoMXAcZZpjF3ierYzYDqHEpQLU9gK6auHIWhB1uthsLfqUAnzGLcBiW5x'
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def text_to_speech(self, text, voice="cixingnansheng"):
        try:
            response = self.client.audio.speech.create(
                model="step-tts-mini", # 追求实时性建议用 mini 版
                voice=voice,
                input=text,
                response_format="mp3" # 常见的格式，Streamlit 易于播放
            )
            # 直接返回二进制内容，方便 Streamlit 处理
            return response.content 
        except Exception as e:
            print(f"TTS Error: {e}")
            return None"""

from pathlib import Path
from openai import OpenAI
 
speech_file_path = r"C:\Users\Jason骆\Desktop\audio_TTS_test\test_interview.mp3" #暂时先这样写吧
 
client = OpenAI(
api_key="6pZ3jWJGHoMXAcZZpjF3ierYzYDqHEpQLU9gK6auHIWhB1uthsLfqUAnzGLcBiW5x",
base_url="https://api.stepfun.com/v1"
)
response = client.audio.speech.create(
model="step-tts-2",
voice="cixingnansheng",
input="我是面试官，请问你是来面试什么职位的呢", #输入的文本
extra_body={
  "volume":1.0 ,# volume 在拓展参数里
  "voice_label":{
    #"language": "粤语", # 可选：语言
    #"emotion": "严肃", # 可选：情感
    "style": "严肃" # 可选：说话语速
  },
  "pronunciation_map":{
    "tone":[
      "阿胶/e1胶",
      "扁舟/偏舟",
      "LOL/laugh out loudly"
      ]
  }
}
)
response.stream_to_file(speech_file_path)
 