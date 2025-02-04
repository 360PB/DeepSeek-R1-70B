import os
import time
import gradio as gr
from openai import OpenAI

def format_time(seconds_float):
    total_seconds = int(round(seconds_float))
    hours = total_seconds // 3600
    remaining_seconds = total_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

DESCRIPTION = '''
<div class="header-container">
    <div class="logo-title">
        <img src="https://cdn.deepseek.com/platform/favicon.png" alt="DeepSeek Logo" class="brand-logo">
        <div class="title-group">
            <h1 class="main-title">我是 DeepSeek，很高兴见到你！</h1>
            <p class="sub-title">我可以帮你写代码、读文件、写作各种创意内容，请把你的任务交给我吧~</p>
            <p class="note">注: 此页面使用 Groq API。<a href="https://console.groq.com/keys" target="_blank">点这里申请 API Key</a></p>
        </div>
    </div>
    <div class="divider-line"></div>
</div>
'''

CSS = """
:root {
    --primary-color: #6366f1;
    --secondary-color: #4f46e5;
    --background: #f8fafc;
    --text-primary: #1e293b;
    --border-color: #e2e8f0;
}

body {
    background: var(--background) !important;
    font-family: system-ui, -apple-system, sans-serif !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 2rem auto !important;
    background: white !important;
    border-radius: 1rem !important;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    padding: 2rem !important;
}

.header-container {
    margin-bottom: 2rem;
}

.logo-title {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}

.brand-logo {
    height: 3.5rem;
    width: auto;
}

.title-group {
    flex-grow: 1;
}

.main-title {
    font-size: 1.75rem;
    color: var(--text-primary);
    margin: 0;
    font-weight: 600;
}

.sub-title {
    color: #64748b;
    margin: 0.25rem 0 0;
}

.divider-line {
    height: 1px;
    background: var(--border-color);
    margin: 1.5rem 0;
}

#chatbot {
    height: 600px !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 0.75rem !important;
    background: white !important;
    overflow-y: auto !important;
}

.message {
    padding: 1rem !important;
    border-radius: 0.5rem !important;
    margin: 0.5rem 0 !important;
}

.user-message {
    background: var(--primary-color) !important;
    color: white !important;
}

.bot-message {
    background: #f1f5f9 !important;
}

.input-row {
    position: relative !important;
    margin-top: 1rem !important;
}

.input-box {
    width: 100% !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 0.75rem !important;
    padding: 0.75rem !important;
    padding-right: 120px !important;
    min-height: 48px !important;
    line-height: 1.5 !important;
    font-size: 1rem !important;
    resize: none !important;
    background: white !important;
}

.send-btn {
    position: absolute !important;
    right: 0.5rem !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    background: var(--primary-color) !important;
    color: white !important;
    border: none !important;
    border-radius: 0.5rem !important;
    padding: 0.5rem 1.5rem !important;
    height: 36px !important;
    min-width: 80px !important;
    font-size: 1rem !important;
    cursor: pointer !important;
    transition: background-color 0.2s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.send-btn:hover {
    background: var(--secondary-color) !important;
}

.error-message {
    color: #ef4444 !important;
    padding: 1rem !important;
    border: 1px solid #ef4444 !important;
    border-radius: 0.5rem !important;
    margin: 1rem 0 !important;
}

.thinking-container {
    border-left: 3px solid var(--primary-color) !important;
    background: #f8fafc !important;
    padding: 1rem !important;
    margin: 1rem 0 !important;
    border-radius: 0.5rem !important;
}
"""

class ParserState:
    __slots__ = ['answer', 'thought', 'in_think', 'start_time', 'last_pos', 'total_think_time']
    def __init__(self):
        self.answer = ""
        self.thought = ""
        self.in_think = False
        self.start_time = 0
        self.last_pos = 0
        self.total_think_time = 0.0

def parse_response(text, state):
    buffer = text[state.last_pos:]
    state.last_pos = len(text)
    
    while buffer:
        if not state.in_think:
            think_start = buffer.find('<think>')
            if think_start != -1:
                state.answer += buffer[:think_start]
                state.in_think = True
                state.start_time = time.perf_counter()
                buffer = buffer[think_start + 7:]
            else:
                state.answer += buffer
                break
        else:
            think_end = buffer.find('</think>')
            if think_end != -1:
                state.thought += buffer[:think_end]
                duration = time.perf_counter() - state.start_time
                state.total_think_time += duration
                state.in_think = False
                buffer = buffer[think_end + 8:]
            else:
                state.thought += buffer
                break
    
    elapsed = time.perf_counter() - state.start_time if state.in_think else 0
    return state, elapsed

def format_response(state, elapsed):
    answer_part = state.answer.replace('<think>', '').replace('</think>', '')
    collapsible = []

    if state.thought or state.in_think:
        if state.in_think:
            total_elapsed = state.total_think_time + elapsed
            formatted_time = format_time(total_elapsed)
            status = f"🤔 思考中... ({formatted_time})"
        else:
            formatted_time = format_time(state.total_think_time)
            status = f"💡 推理过程 ({formatted_time})"
            
        collapsible.append(
            f"<details><summary>{status}</summary>"
            f"<div class='thinking-container'>{state.thought}</div></details>"
        )

    return collapsible, answer_part

def user(message, history):
    if not message.strip():
        raise gr.Error("请输入有效的消息")
    history = history or []
    return "", history + [{"role": "user", "content": message}]

def generate_response(history, temperature, system_prompt, max_tokens, active_gen):
    if not history:
        return history
        
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Only include the essential fields in the messages
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    state = ParserState()
    full_response = ""
    assistant_message = {"role": "assistant", "content": ""}
    history.append(assistant_message)

    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        )
        
        stream = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=messages,
            temperature=temperature,
            stream=True
        )

        for chunk in stream:
            if not active_gen[0]:
                break
                
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                state, elapsed = parse_response(full_response, state)
                collapsible, answer_part = format_response(state, elapsed)
                assistant_message["content"] = "\n\n".join(collapsible + [answer_part])
                yield history

        if active_gen[0]:
            state, elapsed = parse_response(full_response, state)
            collapsible, answer_part = format_response(state, elapsed)
            assistant_message["content"] = "\n\n".join(collapsible + [answer_part])
            yield history

    except Exception as e:
        error_msg = str(e)
        if "GROQ_API_KEY" not in os.environ:
            error_msg = "请设置 GROQ_API_KEY 环境变量"
        assistant_message["content"] = f"<div class='error-message'>⚠️ {error_msg}</div>"
        yield history
    finally:
        active_gen[0] = False

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(DESCRIPTION)
    active_gen = gr.State([False])
    
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_label=False,
        render_markdown=True,
        avatar_images=(None, "https://cdn.deepseek.com/platform/favicon.png"),
        type="messages"
    )

    with gr.Row(elem_classes="input-row"):
        msg = gr.Textbox(
            label="",
            placeholder="输入您的问题...",
            lines=1,
            max_lines=5,
            container=False,
            elem_classes="input-box"
        )
        submit_btn = gr.Button("发送", elem_classes="send-btn")

    with gr.Accordion("⚙️ 高级设置", open=False):
        system_prompt = gr.Textbox(
            label="系统提示词",
            placeholder="输入系统提示词以设定AI助手的行为方式...",
            value="""你是DeepSeek-R1，一个由深度求索公司开发的智能助手，你会以诚实专业的态度帮助用户，用中文回答问题。你会严格遵循以下要求：            
            ### 基本准则
            - 用与用户相同的语言回复
            - 友好、简洁、相关
            - 避免重复内容或偏离主题
            - 拒绝不道德或有害请求
            - 不提供时效性强或需要实时更新的信息
            - 不编造未知信息
            - 代码用Markdown格式
            - 数学公式用LaTeX
            ### 安全合规
            - 禁止讨论政治、领导人、政党
            - 不提供医疗、法律、金融建议
            - 不参与涉及暴力、欺诈等非法场景
            - 不生成歧视性内容
            - 遇到危险请求时明确拒绝
            ### 能力说明
            - 数学计算需分步展示过程
            - 代码问题先解释思路再写代码
            - 文件处理需用户提供内容
            - 联网搜索需具体查询词
            - 图片生成需转换为文生图提示词
            ### 交互规范
            - 不主动结束对话
            - 不解释自身局限性
            - 不讨论内部工作原理
            - 不重复用户问题
            - 遇到无法处理的情况建议转换话题
            不要输出内部推理过程，最终回复要简洁自然。""",
            lines=3
        )
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            label="温度 (创造性)",
            info="较高的值会产生更有创意的回复，较低的值会产生更确定的回复"
        )
        
        max_tokens = gr.Slider(
            minimum=512,
            maximum=128000,
            value=128000,
            step=64,
            label="最大生成长度",
            info="控制生成内容的最大token数量"
        )        

    msg.submit(
        user,
        [msg, chatbot],
        [msg, chatbot],
        queue=False
    ).then(
        lambda: [True],
        outputs=active_gen
    ).then(
        generate_response,
        [chatbot, temperature, system_prompt, max_tokens, active_gen],
        chatbot
    )

    submit_btn.click(
        user,
        [msg, chatbot],
        [msg, chatbot],
        queue=False
    ).then(
        lambda: [True],
        outputs=active_gen
    ).then(
        generate_response,
        [chatbot, temperature, system_prompt, max_tokens, active_gen],
        chatbot
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )