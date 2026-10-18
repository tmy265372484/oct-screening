"""Health guidance: rule-based safety checks with an optional DeepSeek LLM fallback."""
import os
from html import escape

import requests

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

EMERGENCY_WORDS = ("突然", "骤降", "看不见", "黑幕", "幕布", "闪光", "剧痛",
                   "剧烈眼痛", "外伤", "化学", "飞蚊", "失明", "暴盲", "遮住")

SYSTEM_PROMPT = (
    "你叫眸析，是OCT工作站里的眼健康科普助手。你不是医生，也不要假装是真人；但说话要有真实聊天的节奏，"
    "自然、耐心、有分寸，不像在生成报告或念客服话术。"
    "先接住用户刚刚说的重点，再直接回答。句子长短自然，能两句说清就别拉长；通常控制在一小段。"
    "不要每轮都问候、复述问题、总结或加‘希望能帮到你’；不要说‘作为AI/大模型’，也少用‘值得注意的是’、"
    "‘建议您’、‘请知悉’这类模板开头。不要为了显得亲切硬加共情。"
    "不必每次都追问；只有确实需要补充信息时，才用日常说法问一个问题。除非用户要求，不要强行列清单。"
    "语气示例（只学表达方式，不要照抄）：用户问‘复诊要准备什么？’时，可以说‘把之前的OCT报告和现在用的眼药水带上就行。"
    "最近视力或不适有变化的话，简单记下来，医生问起时会更好描述。’用户问‘这个分数很高，是不是确诊了？’时，可以说‘还不能这么理解。"
    "这个分数是模型对某个类别的匹配程度，不是患病概率；结果要结合完整检查由眼科医生判断。’"
    "你可以解释一般眼科概念、筛查报告术语、就诊准备和常见治疗方向；不能诊断个人、开药、给剂量或制定治疗方案。"
    "你看不到用户的OCT图像或本地病例，不要声称已经查看影像或确认病灶；只有当这会影响回答时，才自然说明这一点。"
    "不要建议用户自行购药、停药或改变治疗。若用户描述突然视力下降、幕布样遮挡、新发闪光、飞蚊突然明显增多、"
    "剧烈眼痛或眼外伤，直接、平静地建议尽快前往眼科急诊，不要再作推测。"
    "若问题与眼科无关，简短回应后自然引回眼健康话题。"
)


def is_emergency(text):
    return any(w in text for w in EMERGENCY_WORDS)


def ask_deepseek(api_key, messages, timeout=25):
    response = requests.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": DEEPSEEK_MODEL, "messages": messages, "temperature": 0.55, "max_tokens": 450},
        timeout=timeout,
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"].strip()
    if not answer:
        raise ValueError("The model returned an empty answer")
    return answer


def get_api_key(st):
    """Read the server-side DeepSeek key without exposing it to the browser."""
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    try:
        return st.secrets.get("DEEPSEEK_API_KEY", "").strip()
    except Exception:
        return ""


TOPICS = {
"CNV": "脉络膜新生血管是眼底异常长出的新生血管。医生会结合 OCT、视力和其他检查，判断是否需要抗 VEGF 等治疗并安排复查；单靠模型分类不能决定方案。",
"DME": "糖尿病黄斑水肿需要结合眼底检查和血糖等情况一起评估。有些人会考虑抗 VEGF 等治疗，也有人先处理全身因素，具体要由医生看完整检查后决定。",
"DRUSEN": "玻璃膜疣是黄斑区域常见的一种沉积表现，是否需要处理要看数量、位置和视网膜情况。发现玻璃膜疣不等于一定要打针或吃补充剂，按医生安排复查就好。",
"NORMAL": "模型把这张切片分到‘正常’类，只表示它更像正常样本，不能排除其他眼病。如果你仍有视物模糊、变形或不舒服，还是需要做完整检查。"
}


def reply(text, history=(), category=None):
    # Never infer absence of red flags from keywords or a NORMAL image.
    if is_emergency(text):
        return ("如果这是现在正在发生的突然视力下降、黑影像幕布挡住视野，或剧烈眼痛，建议马上去眼科急诊，别等线上回复。"
                "如果是以前发生的症状，告诉我现在还在不在就好。")
    if any(w in text for w in ("治疗", "药", "注射", "手术", "方案")):
        detail = TOPICS.get(category, "眼病的处理取决于病因和完整检查。有些情况需要观察随访，有些需要药物、激光或手术评估。")
        return detail+"\n\n具体怎么治还得看完整检查，先别自行改药。你是想了解报告里的哪一类情况？"
    names = {"CNV": ("CNV", "新生血管"), "DME": ("DME", "水肿"), "DRUSEN": ("DRUSEN", "玻璃膜疣"), "NORMAL": ("NORMAL", "正常")}
    for key, terms in names.items():
        if any(w.lower() in text.lower() for w in terms):
            return TOPICS[key]
    if any(w in text for w in ("报告", "结果", "热图", "分数")):
        detail = TOPICS.get(category)
        if detail:
            return detail+"分数是模型的类别匹配程度，不是患病概率；关注图也不等于病灶边界。"
        return "你把报告里的分类名称或分数发我，我可以帮你解释。分数不是患病概率，关注图也不等于病灶边界。"
    if any(w in text for w in ("就诊", "准备", "检查准备", "复诊")):
        return "把之前的 OCT 报告和现在用的眼药水带上就行。最近视力或不适有变化的话，简单记下来，医生问起时会更好描述。"
    return "嗯，先从最困扰你的地方说起就好。什么时候开始的？是单眼还是双眼？我可以帮你把就诊时要说的重点理一理。"


def generate_answer(text, history, category, api_key):
    """急症先拦截；否则调 DeepSeek，失败或无 key 时回退本地规则。"""
    if is_emergency(text):
        return reply(text, history, category)
    if api_key:
        try:
            recent_history = [
                {"role": item["role"], "content": str(item["content"])[:2000]}
                for item in history[-8:]
                if item.get("role") in ("user", "assistant") and item.get("content")
            ]
            # Keep local case labels and OCT images out of third-party model requests.
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent_history
            messages.append({"role": "user", "content": text[:1000]})
            return ask_deepseek(api_key, messages)
        except Exception:
            return reply(text, history, category) + "\n\n我这会儿没连上在线答疑，先给你个简短参考。"
    return reply(text, history, category)


def consultation_page(banner):
    import streamlit as st
    from case_library import list_cases
    api_key = get_api_key(st)
    banner("健康咨询", "理解筛查提示 · 整理症状 · 准备就诊")
    left, right = st.columns([0.95, 2.5], gap="large")
    with left, st.container(border=True):
        st.markdown("### 眸析咨询")
        st.caption("眼科科普 · 报告理解 · 就诊准备")
        st.badge("在线答疑" if api_key else "本地知识", color="green" if api_key else "gray")
        rows = list_cases()
        lookup = {r["id"]: r for r in rows}
        chosen = st.selectbox("关联案例（可选）", ["不关联"]+list(lookup),
                              format_func=lambda x: x if x == "不关联" else lookup[x]["source"]+" · "+lookup[x]["category"],
                              key="consult_case")
        category = lookup[chosen]["category"] if chosen in lookup else None
        st.caption("关联案例仅在本机用于提示；在线问答不会发送病例分类或影像。")
        st.warning("突然视力下降、黑影遮挡或剧烈眼痛，请尽快就医。")
        if api_key:
            st.caption("在线答疑会把聊天文字发送给 DeepSeek；请勿填写姓名、电话等身份信息。")
        else:
            st.caption("尚未配置在线服务，当前使用本地知识提示。")
        st.markdown("[眼健康知识 · 美国国家眼科研究所](https://www.nei.nih.gov/learn-about-eye-health)")
    key = "health_messages_"+chosen
    st.session_state.setdefault(key, [])
    with right:
        messages = st.session_state[key]
        # Bounded session-only conversation, never written to case database.
        st.session_state[key] = messages[-40:]
        with st.container(height=350, border=True, key="consult-conversation"):
            if not st.session_state[key]:
                greeting = "你好，我在。想先看看筛查报告里的分类，还是聊聊最近的眼部不适？"
                st.markdown(_chat_bubble("assistant", greeting), unsafe_allow_html=True)
            else:
                for message in st.session_state[key]:
                    st.markdown(_chat_bubble(message["role"], message["content"]), unsafe_allow_html=True)
            # 最后一条是用户提问且尚未回答：生成并显示自然的简短回复
            if st.session_state[key] and st.session_state[key][-1]["role"] == "user":
                last_text = st.session_state[key][-1]["content"]
                with st.spinner("稍等，我整理一下…"):
                    answer = generate_answer(last_text, st.session_state[key][:-1], category, api_key)
                st.session_state[key].append({"role": "assistant", "content": answer})
                st.rerun()
        with st.container(horizontal=True):
            topics = ["报告里的分类是什么意思？", "这个分数代表什么？", "复诊前要准备什么？"]
            for i, label in enumerate(topics):
                if st.button(label, key="consult_topic_"+str(i)):
                    st.session_state[key].append({"role": "user", "content": label})
                    st.rerun()
        prompt = st.chat_input("发条消息…", max_chars=1000)
        if prompt:
            st.session_state[key].append({"role": "user", "content": prompt})
            st.rerun()
        with st.container(horizontal=True):
            role_names = {"user": "我", "assistant": "眸析"}
            transcript = "\n\n".join(role_names.get(m["role"], "对话")+"： "+m["content"] for m in st.session_state[key])
            st.download_button("导出对话", transcript,
                               file_name="健康咨询记录.txt", mime="text/plain")
            if st.button("清空对话"):
                st.session_state[key] = []
                st.rerun()


def _chat_bubble(role, content):
    """Render escaped plain text in compact, directional chat bubbles."""
    role = "assistant" if role == "assistant" else "user"
    label, avatar = ("眸析", "眸") if role == "assistant" else ("我", "我")
    safe_content = escape(str(content)).replace("\n", "<br>")
    return (
        f'<div class="chat-row chat-row--{role}" role="group" aria-label="{label}的消息">'
        f'<span class="chat-avatar" aria-hidden="true">{avatar}</span>'
        f'<div class="chat-bubble">{safe_content}</div></div>'
    )
