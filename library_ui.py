"""Persistent archive with searchable gallery, real images and case details."""
from html import escape
import streamlit as st
from case_library import list_cases, load_case, save_case, update_note, delete_case
from config import CLASS_CN
from viewer_features import annotation_editor, annotated_image, validate_annotations

def library_page(banner,canvas,downloads,image_uri):
    banner("案例库", "每一次分析都有记录 · 影像、结果与备注完整回看")
    st.caption("案例保存在本机，刷新页面后仍可查看。当前为单机共享案例库，请勿在公共电脑导入可识别个人身份的资料。")
    search,filtercol=st.columns([3,1])
    query=search.text_input("搜索案例",placeholder="输入文件名、案例编号或备注",key="library_search")
    category=filtercol.selectbox("分类筛选",["全部","CNV","DME","DRUSEN","NORMAL"])
    rows=list_cases(query,category)
    st.caption(f"找到 {len(rows)} 个案例")
    selected=st.session_state.get("library_selected")
    record=load_case(selected) if selected else None
    if record:
        if st.button("返回案例列表",icon=":material/arrow_back:"):
            st.session_state.pop("library_selected",None)
            st.rerun()
        st.subheader(record["source_name"])
        st.caption(record["case_id"]+" · "+record["created_at"])
        images,details=st.columns([3,1.35],gap="large")
        with images:
            if record.get("annotations"):
                a,b,c=st.columns(3)
                with a: canvas(record["original"],"原始影像")
                with b: canvas(annotated_image(record["original"],record["annotations"],record.get("source_size")),"用户标注")
                with c:
                    if record["overlay"] is not None: canvas(record["overlay"],"AI 关注区域")
                    else: st.info("本次分析没有可用关注图。")
            else:
                a,b=st.columns(2)
                with a: canvas(record["original"],"原始影像")
                with b:
                    if record["overlay"] is not None: canvas(record["overlay"],"已保存的关注区域")
                    else: st.info("本次分析没有可用关注图。")
            if record.get("slice_total",1)>1:
                st.caption(f"序列 {record.get('series_id','未记录')} · 第 {record.get('slice_index','?')} / {record.get('slice_total')} 张")
            st.caption("展示保存时的影像、用户标注与关注叠加图，不会重新运行模型。")
            with st.expander("影像标注与测量", expanded=bool(record.get("annotations"))):
                st.caption("标注用于阅片记录，距离以输入影像像素 px 计；没有设备像素间距时不换算为 μm。")
                changed=annotation_editor(record,key="library_marks_"+selected)
                if changed is not None:
                    record["annotations"]=validate_annotations(changed,record["original"].size)
                    try:
                        save_case(record)
                        st.success("标注已保存在本机案例库。")
                        st.rerun()
                    except Exception:
                        st.error("标注暂未保存，请检查本机磁盘空间或权限。")
        with details,st.container(border=True):
            st.badge("待专业人员复核",color="orange")
            st.subheader(record["prediction"]["predicted_label_cn"])
            st.metric("模型匹配分数",f'{record["prediction"]["model_score"]:.2%}')
            for name,value in sorted(record["prediction"]["probabilities"].items(),key=lambda x:x[1],reverse=True):
                st.progress(float(value),text=f"{CLASS_CN[name]} · {value:.2%}")
            note=st.text_area("案例备注",record["note"],max_chars=1000,key="archive_note_"+selected)
            if st.button("保存备注",icon=":material/save:"):
                update_note(selected,note)
                record["note"]=note
                active=st.session_state.get("active_record")
                if active and active["case_id"]==selected:
                    active["note"]=note
                    st.session_state["note_"+selected]=note
                st.success("备注已保存")
            downloads(record,"library_"+selected)
        with st.expander("来源与记录管理"):
            st.write("输入来源："+record.get("input_kind","未记录"))
            st.code(record.get("source_sha256",""),language=None)
            st.caption("图像指纹用于确认记录对应的输入文件。")
            confirm=st.checkbox("确认永久删除这个本地案例",key="delete_confirm_"+selected)
            if st.button("删除案例",disabled=not confirm,key="delete_"+selected):
                delete_case(selected)
                st.session_state.pop("library_selected",None)
                st.rerun()
        return
    if not rows:
        st.info("暂无匹配案例。完成一次影像分析后会自动入库。")
        return
    page_count=(len(rows)+8)//9
    page=st.number_input("列表页码",min_value=1,max_value=page_count,value=1,step=1,width=180) if page_count>1 else 1
    visible=rows[(page-1)*9:page*9]
    cols=st.columns(3,gap="large")
    for i,row in enumerate(visible):
        record=load_case(row["id"])
        if record is None: continue
        with cols[i%3],st.container(border=True):
            st.html(f'<div class="case-thumb"><img alt="案例原图" src="{image_uri(record["original"])}"></div>')
            st.markdown("**"+escape(row["source"])+"**")
            st.caption(row["created"][:19].replace("T"," "))
            st.badge(CLASS_CN[row["category"]],color="blue")
            st.caption("模型匹配 "+f'{record["prediction"]["model_score"]:.2%}'+" · 待复核")
            if st.button("查看案例详情",key="case_"+row["id"],width="stretch"):
                st.session_state["library_selected"]=row["id"]
                st.rerun()
