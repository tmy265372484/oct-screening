"""Shared review controls inside the existing workstation and case panels."""
import sqlite3
from case_library import review_history, save_review


def review_panel(record, prefix):
    import streamlit as st
    history = review_history(record['case_id'])
    record['reviews'] = history
    latest = history[-1] if history else None
    st.badge('已记录人工复核' if latest else '待专业人员复核', color='green' if latest else 'orange')
    with st.expander('人工复核与记录', expanded=False):
        st.caption('模型提示与人工结论分别保存。复核者标识由本机填写，不是身份认证或电子签名。')
        if latest:
            st.write(f"第 {latest['revision']} 次复核 · {latest['decision']} · {latest.get('category') or '待补充检查'}")
            st.caption(latest['reviewed_at'])
        with st.form(prefix+'_review_'+record['case_id']):
            reviewer = st.text_input('复核者标识', max_chars=80)
            decision = st.selectbox('复核结论', ['确认模型分类', '修正分类', '暂不能判断'])
            category = st.selectbox('修正后的分类（仅修正时使用）', ['CNV','DME','DRUSEN','NORMAL','其他/超出模型范围'])
            rationale = st.text_area('复核依据与需补充的检查', max_chars=2000)
            confirmed = st.checkbox('我已查看影像，并对以上人工复核内容负责')
            submitted = st.form_submit_button('保存复核记录')
        if submitted:
            if not confirmed:
                st.error('请先确认已查看影像。')
            else:
                try:
                    save_review(record['case_id'], reviewer, decision, category, rationale, len(history))
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
                except (OSError, sqlite3.Error):
                    st.error('复核未保存，请检查磁盘。')
        for item in reversed(history):
            st.write(f"#{item['revision']} · {item['reviewed_at']} · {item['reviewer']} · {item['decision']} · {item.get('category') or '未定'}")
            st.text(item['rationale'])
