# llm_agent.py
import os
import re
from typing import Generator, Dict, List, Any
from openai import OpenAI

# API配置
api_key = os.getenv("DEEPSEEK_API_KEY", "sk-4922e9d067814b109ee7663fffee442e")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def clean_text(text: str, max_len: int = 1500) -> str:
    
    if not text or not text.strip():
        return ""
    
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len]
    
    
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    return text

def manage_history(history: List[Dict], max_length: int = 6) -> List[Dict]:
    """管理对话历史，保持合理长度"""
    if len(history) <= max_length:
        return history.copy()
    
    new_history = []
    for msg in history:
        if msg.get("role") == "system":
            new_history.append(msg)
            break
    
    recent_count = max_length - len(new_history)
    return new_history + history[-recent_count:]

def llm_stream_chat(
    history: List[Dict],
    user_input: str,
    system_prompt: str = "",
    model_name: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    top_p: float = 0.9,
    **extra_params
) -> Generator[str, None, None]:
    """
    核心LLM函数：文字入，文字出
    参数全部从app.py传入
    """
    # 清理输入
    user_input = clean_text(user_input)
    if not user_input:
        yield "请输入有效内容"
        return
    
    # 准备历史记录
    history = manage_history(history)
    messages = history.copy()
    
    # 添加系统提示（如果有）
    if system_prompt and not any(msg.get("role") == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": user_input})
 
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **extra_params
        )
        
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"抱歉，系统出了点问题: {str(e)}"

def get_full_response(history: List[Dict], user_input: str, system_prompt: str = "") -> str:
    """获取完整响应（非流式）"""
    response = ""
    for chunk in llm_stream_chat(history, user_input, system_prompt):
        response += chunk
    return response

def evaluate_answer(question: str, answer: str) -> Dict[str, Any]:
    """评估回答质量"""
    score = 3.0
    
    # 基于长度
    if len(answer) > 100:
        score += 0.5
    elif len(answer) < 30:
        score -= 1.0
    
    # 基于内容
    tech_words = ["Python", "Java", "算法", "框架", "数据库", "优化", "模型"]
    for word in tech_words:
        if word in answer:
            score += 0.2
            break
    
    vague_words = ["不太清楚", "不了解", "大概", "可能"]
    for word in vague_words:
        if word in answer:
            score -= 0.3
            break
    
    # 限制范围
    score = max(1.0, min(5.0, score))
    
    # 反馈
    if score >= 4.0:
        feedback = "回答详细，技术点清晰"
    elif score >= 3.0:
        feedback = "回答基本完整"
    else:
        feedback = "可以更详细一些"
    
    return {"score": round(score, 1), "feedback": feedback}

def get_summary(position: str, question_count: int, scores: List[float]) -> str:
    """生成面试总结"""
    if not scores:
        return "面试尚未开始"
    
    avg_score = sum(scores) / len(scores)
    
    summary = f"""
面试结束！
职位：{position}
问题数量：{question_count}
平均得分：{avg_score:.1f}/5.0

{"表现优秀，专业扎实" if avg_score >= 4.0 else "基本合格，仍有提升空间"}
"""
    return summary.strip()
