"""Deterministic educational guidance, explicitly not an LLM or diagnosis."""
TOPICS = {
"CNV": "脉络膜新生血管指异常新生血管。确诊后，眼科医生常会评估是否适合抗 VEGF 等治疗，并安排影像随访。是否需要治疗及具体方案取决于病因、视力和完整检查，不能由本系统决定。",
"DME": "糖尿病相关黄斑水肿需要眼科检查，并结合血糖、血压等全身情况管理。医生可能评估抗 VEGF、其他眼内治疗或特定情况下的激光治疗。不能只凭模型标签确定方案。",
"DRUSEN": "玻璃膜疣是眼底的一类沉积表现，是否需要随访或其他干预取决于黄斑状态与疾病分期。并不是出现玻璃膜疣就需要注射或服用补充剂，需由眼科医生评估。",
"NORMAL": "正常类别表示这张切片更接近模型的正常样本，不能排除其他眼病。如果仍有视物模糊、变形或不适，应进行完整眼科检查。"
}
def reply(text, history=(), category=None):
    # Never infer absence of red flags from keywords or a NORMAL image.
    if any(w in text for w in ("突然","骤降","看不见","黑幕","幕布","闪光","剧痛","剧烈眼痛","外伤","化学","飞蚊")):
        return ("你提到了可能需要及时处理的症状。若正在出现突然视力下降、幕布样遮挡、新发闪光或飞蚊明显增多，或剧烈眼痛、眼外伤，请尽快前往眼科急诊，不要等待在线判断。"
                "\n\n这是关键词触发的提醒，不是确诊；如果是在描述既往或否认这些症状，可以说明发生时间和目前情况。")
    if any(w in text for w in ("治疗","药","注射","手术","方案")):
        detail = TOPICS.get(category,"眼病的处理取决于病因和完整检查。有些情况需要观察随访，有些需要药物、激光或手术评估。")
        return detail+"\n\n以上是一般治疗方向，不是个人处方。请勿自行使用眼药、激素或停用原有治疗。你可以记录起病时间、单眼/双眼、既往病史与目前用药，带上原始检查就诊。"
    names = {"CNV":("CNV","新生血管"),"DME":("DME","水肿"),"DRUSEN":("DRUSEN","玻璃膜疣"),"NORMAL":("NORMAL","正常")}
    for key,terms in names.items():
        if any(w.lower() in text.lower() for w in terms):
            return TOPICS[key]
    if any(w in text for w in ("报告","结果","热图","分数")):
        return (TOPICS.get(category,"请先在左侧选择已保存的案例，我可以展示该分类的一般解释。")
                +"\n\n分数不是患病概率，关注图不是病灶边界。本地问答不重新读取或诊断影像。")
    if any(w in text for w in ("就诊","准备","检查准备","复诊")):
        return "建议准备：症状开始时间和变化、单眼或双眼、既往眼病/糖尿病等病史、正在使用的药物，以及原始 OCT 和既往检查。复诊时间应遵循接诊医生安排；新发或明显加重的症状需要及时就诊。"
    return ("为了帮助你整理就诊信息，请描述：①主要不适是什么；②什么时候开始、是否加重；③单眼还是双眼；④是否伴眼痛、闪光、黑影或视力突然下降。"
            "\n\n当前是本地规则问答，不能像医生一样全面理解自由描述或判断疾病。你也可以问「解释报告」「治疗方向有哪些」「就诊需要准备什么」。")

def consultation_page(banner):
    import streamlit as st
    from case_library import list_cases
    banner("健康咨询", "理解筛查提示 · 整理症状 · 准备就诊")
    left,right = st.columns([1,2.5],gap="large")
    with left, st.container(border=True):
        st.markdown("### 眸析健康助手")
        st.badge("本地问答引导",color="blue")
        st.caption("非医生 · 未接入大模型。提供一般知识与就诊引导，不开具个体治疗方案。")
        rows=list_cases()
        lookup={r["id"]:r for r in rows}
        chosen=st.selectbox("关联案例（可选）",["不关联"]+list(lookup),
                            format_func=lambda x: x if x=="不关联" else lookup[x]["source"]+" · "+lookup[x]["category"],
                            key="consult_case")
        category=lookup[chosen]["category"] if chosen in lookup else None
        st.caption("关联仅用于解释分类，不会将模型分类当作确诊。")
        st.warning("突然视力下降、幕布样遮挡或剧烈眼痛，请及时前往眼科急诊。")
        st.markdown("[眼健康知识 · 美国国家眼科研究所](https://www.nei.nih.gov/learn-about-eye-health)")
    key="health_messages_"+chosen
    st.session_state.setdefault(key,[])
    with right:
        st.info("你好，我可以帮你解释筛查提示和整理就诊问题。请不要填写姓名、电话或身份证号。")
        with st.container(horizontal=True):
            for i,label in enumerate(["解释报告","治疗方向有哪些","就诊需要准备什么"]):
                if st.button(label,key="topic_"+str(i)):
                    st.session_state[key].extend([{"role":"user","content":label},{"role":"assistant","content":reply(label,category=category)}])
        messages=st.session_state[key]
        # Bounded session-only conversation, never written to case database.
        st.session_state[key]=messages[-40:]
        with st.container(height=440,border=True):
            for message in st.session_state[key]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        prompt=st.chat_input("描述症状或询问报告含义…",max_chars=1000)
        if prompt:
            st.session_state[key].extend([{"role":"user","content":prompt},{"role":"assistant","content":reply(prompt,messages,category)}])
            st.rerun()
        with st.container(horizontal=True):
            st.download_button("导出咨询记录","\n\n".join(m["role"]+"： "+m["content"] for m in st.session_state[key]),
                               file_name="健康咨询记录.txt",mime="text/plain")
            if st.button("清空本段咨询"):
                st.session_state[key]=[]
                st.rerun()
