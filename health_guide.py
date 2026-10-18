"""Health guidance: rule-based safety checks with an optional DeepSeek LLM fallback."""
import requests

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

EMERGENCY_WORDS = ("突然", "骤降", "看不见", "黑幕", "幕布", "闪光", "剧痛",
                   "剧烈眼痛", "外伤", "化学", "飞蚊", "失明", "暴盲", "遮住")

SYSTEM_PROMPT = (
    "你是「眸析」OCT 视网膜影像筛查工作台的健康咨询助手。"
    "职责：用通俗中文解释眼科/眼底影像相关概念、解释筛查结果含义、整理就诊问题、介绍一般就诊准备与治疗方向。"
    "必须遵守：1) 只做科普与教育引导，绝不提供个体化诊断、治疗方案、药物名称/剂量或处方；"
    "2) 不解读具体医学影像、不声称能诊断疾病；"
    "3) 遇到突然视力下降、幕布样/黑幕遮挡、新发闪光、飞蚊突然增多、剧烈眼痛、眼外伤等急症表现，一律建议尽快前往眼科急诊并停止进一步推测；"
    "4) 不引导用户自行购药、停药或改变现有治疗；5) 回答简洁分点、避免恐吓性语言。"
    "若用户询问与眼科无关的问题，简要回应后引导回眼科健康话题。"
)


def is_emergency(text):
    return any(w in text for w in EMERGENCY_WORDS)


def ask_deepseek(api_key, messages, timeout=40):
    response = requests.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": DEEPSEEK_MODEL, "messages": messages, "temperature": 0.4, "max_tokens": 1000},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


TOPICS = {
"CNV": "脉络膜新生血管指异常新生血管。确诊后，眼科医生常会评估是否适合抗 VEGF 等治疗，并安排影像随访。是否需要治疗及具体方案取决于病因、视力和完整检查，不能由本系统决定。",
"DME": "糖尿病相关黄斑水肿需要眼科检查，并结合血糖、血压等全身情况管理。医生可能评估抗 VEGF、其他眼内治疗或特定情况下的激光治疗。不能只凭模型标签确定方案。",
"DRUSEN": "玻璃膜疣是眼底的一类沉积表现，是否需要随访或其他干预取决于黄斑状态与疾病分期。并不是出现玻璃膜疣就需要注射或服用补充剂，需由眼科医生评估。",
"NORMAL": "正常类别表示这张切片更接近模型的正常样本，不能排除其他眼病。如果仍有视物模糊、变形或不适，应进行完整眼科检查。"
}


def reply(text, history=(), category=None):
    # Never infer absence of red flags from keywords or a NORMAL image.
    if is_emergency(text):
        return ("你提到了可能需要及时处理的症状。若正在出现突然视力下降、幕布样遮挡、新发闪光或飞蚊明显增多，或剧烈眼痛、眼外伤，请尽快前往眼科急诊，不要等待在线判断。"
                "\n\n这是关键词触发的提醒，不是确诊；如果是在描述既往或否认这些症状，可以说明发生时间和目前情况。")
    if any(w in text for w in ("治疗", "药", "注射", "手术", "方案")):
        detail = TOPICS.get(category, "眼病的处理取决于病因和完整检查。有些情况需要观察随访，有些需要药物、激光或手术评估。")
        return detail+"\n\n以上是一般治疗方向，不是个人处方。请勿自行使用眼药、激素或停用原有治疗。你可以记录起病时间、单眼/双眼、既往病史与目前用药，带上原始检查就诊。"
    names = {"CNV": ("CNV", "新生血管"), "DME": ("DME", "水肿"), "DRUSEN": ("DRUSEN", "玻璃膜疣"), "NORMAL": ("NORMAL", "正常")}
    for key, terms in names.items():
        if any(w.lower() in text.lower() for w in terms):
            return TOPICS[key]
    if any(w in text for w in ("报告", "结果", "热图", "分数")):
        return (TOPICS.get(category, "请先在左侧选择已保存的案例，我可以展示该分类的一般解释。")
                +"\n\n分数不是患病概率，关注图不是病灶边界。本地问答不重新读取或诊断影像。")
    if any(w in text for w in ("就诊", "准备", "检查准备", "复诊")):
        return "建议准备：症状开始时间和变化、单眼或双眼、既往眼病/糖尿病等病史、正在使用的药物，以及原始 OCT 和既往检查。复诊时间应遵循接诊医生安排；新发或明显加重的症状需要及时就诊。"
    return ("为了帮助你整理就诊信息，请描述：①主要不适是什么；②什么时候开始、是否加重；③单眼还是双眼；④是否伴眼痛、闪光、黑影或视力突然下降。"
            "\n\n当前是本地规则问答，不能像医生一样全面理解自由描述或判断疾病。你也可以问「解释报告」「治疗方向有哪些」「就诊需要准备什么」。")


def generate_answer(text, history, category, api_key):
    """急症先拦截；否则调 DeepSeek，失败或无 key 时回退本地规则。"""
    if is_emergency(text):
        return reply(text, history, category)
    if api_key:
        sys_prompt = SYSTEM_PROMPT
        if category:
            sys_prompt += f"\n当前用户关联的影像分类为 {category}（{TOPICS.get(category, '')}）。可结合该类别的通用知识回答，但不要据此判断该用户患病。"
        try:
            return ask_deepseek(api_key, [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": text}])
        except Exception:
            return reply(text, history, category) + "\n\n（AI 服务暂时不可用，已切换为本地规则回答。）"
    return reply(text, history, category)


def consultation_page(banner):
    import streamlit as st
    from case_library import list_cases
    banner("健康咨询", "理解筛查提示 · 整理症状 · 准备就诊")
    left, right = st.columns([1, 2.5], gap="large")
    with left, st.container(border=True):
        st.markdown("### 眸析健康助手")
        st.badge("AI 健康助手", color="blue")
        st.caption("AI 辅助 · 仅供参考。提供一般知识与就诊引导，不开具个体治疗方案。")
        rows = list_cases()
        lookup = {r["id"]: r for r in rows}
        chosen = st.selectbox("关联案例（可选）", ["不关联"]+list(lookup),
                              format_func=lambda x: x if x == "不关联" else lookup[x]["source"]+" · "+lookup[x]["category"],
                              key="consult_case")
        category = lookup[chosen]["category"] if chosen in lookup else None
        st.caption("关联仅用于解释分类，不会将模型分类当作确诊。")
        st.warning("突然视力下降、幕布样遮挡或剧烈眼痛，请及时前往眼科急诊。")
        st.markdown("[眼健康知识 · 美国国家眼科研究所](https://www.nei.nih.gov/learn-about-eye-health)")
    try:
        api_key = st.secrets.get("DEEPSEEK_API_KEY", "").strip()
    except Exception:
        api_key = ""
    key = "health_messages_"+chosen
    st.session_state.setdefault(key, [])
    with right:
        if api_key:
            st.info("你好，我可以帮你解释筛查提示和整理就诊问题，回答由 AI 生成、仅供参考。请不要填写姓名、电话或身份证号。")
        else:
            st.info("你好，我可以帮你解释筛查提示和整理就诊问题。当前未配置 AI 接口，使用本地规则回答。请不要填写姓名、电话或身份证号。")
        with st.container(horizontal=True):
            for i, label in enumerate(["解释报告", "治疗方向有哪些", "就诊需要准备什么"]):
                if st.button(label, key="topic_"+str(i)):
                    st.session_state[key].extend([{"role": "user", "content": label}, {"role": "assistant", "content": reply(label, category=category)}])
        messages = st.session_state[key]
        # Bounded session-only conversation, never written to case database.
        st.session_state[key] = messages[-40:]
        with st.container(height=280, border=True):
            for message in st.session_state[key]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            # 最后一条是用户提问且尚未回答：在对话流内显示"正在思考"并生成回答
            if st.session_state[key] and st.session_state[key][-1]["role"] == "user":
                last_text = st.session_state[key][-1]["content"]
                with st.chat_message("assistant"):
                    with st.spinner("正在思考，请稍等…"):
                        answer = generate_answer(last_text, st.session_state[key][:-1], category, api_key)
                st.session_state[key].append({"role": "assistant", "content": answer})
                st.rerun()
        prompt = st.chat_input("描述症状或询问报告含义…", max_chars=1000)
        if prompt:
            st.session_state[key].append({"role": "user", "content": prompt})
            st.rerun()
        with st.container(horizontal=True):
            st.download_button("导出咨询记录", "\n\n".join(m["role"]+"： "+m["content"] for m in st.session_state[key]),
                               file_name="健康咨询记录.txt", mime="text/plain")
            if st.button("清空本段咨询"):
                st.session_state[key] = []
                st.rerun()