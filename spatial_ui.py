"""Spatial-glass landing and shared navigation; native page links preserve sessions."""
import base64
from pathlib import Path
from html import escape
import streamlit as st

ROOT = Path(__file__).resolve().parent
FEATURES = [
    ("workspace", "影像工作台", "IMAGE WORKSPACE", "导入 OCT 影像，发现每一处值得关注的细节。", "view_quilt"),
    ("cases", "案例库", "CASE LIBRARY", "让每一次分析，都有迹可循。", "folder_open"),
    ("compare", "影像对比", "IMAGE COMPARISON", "并排阅片，细察影像差异。", "compare"),
    ("statistics", "统计分析", "ANALYTICS", "从影像结果中，看见全局。", "bar_chart"),
    ("consultation", "健康问诊", "HEALTH CONSULTATION", "理解报告，让问题得到回应。", "forum"),
]

@st.cache_data(show_spinner=False)
def background_css():
    encoded = base64.b64encode((ROOT/"assets/spatial-retina.png").read_bytes()).decode("ascii")
    return '<style>[data-testid="stAppViewContainer"]{background-image:linear-gradient(90deg,rgba(5,10,13,.48),rgba(5,10,13,.16)),url("data:image/png;base64,'+encoded+'")!important;background-size:cover!important;background-position:center!important;background-attachment:fixed!important}</style>'

def install_style():
    st.html(background_css())
    st.html((ROOT/"assets/spatial-glass.css").read_text(encoding="utf-8"))

def glass_card(feature, pages, index, large=False):
    route,title,eyebrow,description,icon=feature
    symbols = {
        "workspace": '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M3 10h18M10 10v10"/>',
        "cases": '<path d="M3 7a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v9H3Z"/>',
        "compare": '<rect x="2" y="5" width="8" height="14" rx="2"/><rect x="14" y="5" width="8" height="14" rx="2"/><path d="M12 2v20"/>',
        "statistics": '<path d="M4 20V11M12 20V4M20 20V8M2 22h20"/>',
        "consultation": '<path d="M21 11a8 8 0 0 1-8 8H7l-4 3V11a8 8 0 0 1 18 0Z"/><path d="M7 10h10M7 14h6"/>',
    }
    symbol = '<svg width="27" height="27" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'+symbols[route]+'</svg>'
    with st.container(key="glass_card_"+route):
        st.html('<div class="glass-card-content '+("card-large" if large else "")+'">'
                '<div class="card-overline"><span>'+str(index).zfill(2)+'</span>'
                +symbol+'</div>'
                '<div class="card-eyebrow">'+eyebrow+'</div><h2>'+title+'</h2><p>'+description+'</p>'
                '</div>')
        st.page_link(pages[route],label="进入"+title+"  ↗",use_container_width=True)

def home(pages):
    with st.container(key="spatial_home"):
        title,cards=st.columns([1,1.9],gap="large",vertical_alignment="center")
        with title:
            st.html('<div class="spatial-hero"><div class="hero-kicker">MOUXI · OCT INTELLIGENCE</div>'
                    '<h1>清晰洞察<br><span>自在眼前。</span></h1>'
                    '<div class="hero-en">SEE BEYOND.<br>LOOK CLOSER.</div>'
                    '<p>在影像之间，看见更多。<br>你的视网膜影像辅助工作空间。</p>'
                    '<div class="hero-details"><span>影像分析</span><i></i><span>案例管理</span><i></i><span>健康咨询</span></div>'
                    '</div>')
        with cards:
            a,b=st.columns([1.25,1],gap="medium")
            with a: glass_card(FEATURES[0],pages,1,True)
            with b: glass_card(FEATURES[1],pages,2,True)
            a,b,c=st.columns(3,gap="medium")
            for col,feature,index in zip((a,b,c),FEATURES[2:],(3,4,5)):
                with col: glass_card(feature,pages,index)
        st.html('<div class="spatial-footer"><span>SPACE. DEPTH. CLARITY.</span>'
                '<span>OCT 影像辅助筛查 · 结果需专业复核</span><span>MOUXI / WORKSPACE</span></div>')

def topbar(pages, home_page, guide_page, current):
    with st.container(key="spatial_topbar"):
        brand,nav=st.columns([1,3],vertical_alignment="center")
        with brand:
            st.html('<div class="spatial-brand"><span class="brand-orbit">◎</span><b>眸析</b>'
                    '<span>MOUXI</span><small>VISION WORKSPACE</small></div>')
        with nav:
            with st.container(horizontal=True,horizontal_alignment="right",gap="small"):
                st.page_link(home_page,label="首页",icon=":material/home:")
                if current.url_path:
                    for route,title,*_ in FEATURES:
                        st.page_link(pages[route],label=title)
                else:
                    st.page_link(guide_page,label="使用帮助",icon=":material/help:")
    if current.url_path:
        st.html('<div class="spatial-section-label">MOUXI / '+escape(current.title)+'</div>')
