"""Local helpers and a lightweight Streamlit canvas for OCT review."""
from __future__ import annotations

import hashlib
from html import escape
import math
import re
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from app_support import ROOT, png_bytes


_ALLOWED_TYPES = {"point", "line", "rect", "text"}
_COLOR = "#ffe27a"


def natural_sort_key(value: str):
    """Sort slice_2 before slice_10 while remaining case-insensitive."""
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", str(value))]


def sequence_id(items) -> str:
    digest = hashlib.sha256()
    for name, payload in items:
        digest.update(str(name).encode("utf-8", "replace"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
        digest.update(b"\0")
    return "SEQ-" + digest.hexdigest()[:12].upper()


def image_data_uri(image: Image.Image) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(png_bytes(image)).decode("ascii")


def validate_annotations(value, image_size: tuple[int, int], limit: int = 250) -> list[dict]:
    """Accept only small, bounded annotation records from the browser component."""
    if isinstance(value, dict):
        value = value.get("annotations", [])
    if not isinstance(value, list):
        return []
    width, height = image_size
    clean = []
    for item in value[:limit]:
        if not isinstance(item, dict) or item.get("type") not in _ALLOWED_TYPES:
            continue
        points = item.get("points")
        expected = 1 if item["type"] in {"point", "text"} else 2
        if not isinstance(points, list) or len(points) != expected:
            continue
        parsed = []
        valid = True
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                valid = False
                break
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError, OverflowError):
                valid = False
                break
            if not math.isfinite(x) or not math.isfinite(y):
                valid = False
                break
            parsed.append([min(max(x, 0.0), max(width - 1, 0)),
                           min(max(y, 0.0), max(height - 1, 0))])
        if not valid:
            continue
        entry = {"id": str(item.get("id", ""))[:48], "type": item["type"], "points": parsed}
        if item["type"] == "text":
            entry["text"] = str(item.get("text", "标注"))[:120]
        clean.append(entry)
    return clean


def _measurement(annotation: dict, shown_size: tuple[int, int], source_size=None) -> str:
    points = annotation.get("points", [])
    scale_x = (source_size[0] / shown_size[0]) if source_size and shown_size[0] else 1.0
    scale_y = (source_size[1] / shown_size[1]) if source_size and shown_size[1] else 1.0
    if annotation.get("type") == "line" and len(points) == 2:
        dx = (points[1][0] - points[0][0]) * scale_x
        dy = (points[1][1] - points[0][1]) * scale_y
        return f"{math.hypot(dx, dy):.1f} px"
    if annotation.get("type") == "rect" and len(points) == 2:
        dx = abs(points[1][0] - points[0][0]) * scale_x
        dy = abs(points[1][1] - points[0][1]) * scale_y
        return f"{dx:.1f} × {dy:.1f} px"
    return ""


def annotated_image(image: Image.Image, annotations, source_size=None) -> Image.Image:
    """Render user marks on a copy for the case gallery and export."""
    base = image.convert("RGB").copy()
    marks = validate_annotations(annotations, base.size)
    if not marks:
        return base
    draw = ImageDraw.Draw(base)
    stroke = max(2, round(max(base.size) / 600))
    radius = max(4, stroke * 2)
    font_size = max(14, round(max(base.size) / 55))
    font = None
    for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    for number, mark in enumerate(marks, start=1):
        points = [(round(p[0]), round(p[1])) for p in mark["points"]]
        kind = mark["type"]
        if kind == "point":
            x, y = points[0]
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=_COLOR, width=stroke)
            label = f"P{number}"
            xy = (x + radius + 3, y - radius - 2)
        elif kind == "line":
            draw.line(points, fill=_COLOR, width=stroke)
            for x, y in points:
                draw.ellipse((x-radius//2, y-radius//2, x+radius//2, y+radius//2), fill=_COLOR)
            label = _measurement(mark, base.size, source_size)
            xy = (points[1][0] + 5, points[1][1] + 5)
        elif kind == "rect":
            draw.rectangle(points, outline=_COLOR, width=stroke)
            label = _measurement(mark, base.size, source_size)
            xy = (min(p[0] for p in points) + 4, min(p[1] for p in points) + 4)
        else:
            label = mark.get("text", "标注")
            xy = points[0]
        if label:
            try:
                box = draw.textbbox(xy, label, font=font, stroke_width=0)
                draw.rectangle((box[0]-3, box[1]-2, box[2]+3, box[3]+2), fill="#101820")
            except AttributeError:
                pass
            draw.text(xy, label, fill=_COLOR, font=font)
    return base


def annotation_editor(record: dict, key: str):
    """Native Streamlit point/line/ROI/text editor with session-local undo/redo."""
    import streamlit as st

    image = record["original"]
    width, height = image.size
    marks = validate_annotations(record.get("annotations", []), image.size)
    source_size = record.get("source_size") or image.size
    st.image(annotated_image(image, marks, source_size), width="stretch")
    st.caption(f"图像坐标为当前预览像素：X 0–{width-1}，Y 0–{height-1}。距离标签按输入影像像素 px 换算；未提供设备像素间距时不显示 μm。")

    history_key = "_annotation_history_" + key
    history = st.session_state.setdefault(history_key, {"undo": [], "redo": [], "last": marks})

    def snapshot(values):
        return [{**item, "points": [list(point) for point in item["points"]]} for item in values]

    if snapshot(history.get("last", [])) != snapshot(marks):
        history["undo"], history["redo"], history["last"] = [], [], snapshot(marks)

    mode = st.selectbox("标注工具", ["点位", "距离测量", "矩形 ROI", "文字"],
                        key=key+"_tool")
    a, b, c, d = st.columns(4, gap="small")
    x1 = int(a.number_input("X1 · px", min_value=0, max_value=max(0, width-1), value=0, step=1, key=key+"_x1"))
    y1 = int(b.number_input("Y1 · px", min_value=0, max_value=max(0, height-1), value=0, step=1, key=key+"_y1"))
    x2_default, y2_default = min(max(width-1, 0), max(1, width//3)), min(max(height-1, 0), max(1, height//3))
    x2, y2 = x2_default, y2_default
    if mode in {"距离测量", "矩形 ROI"}:
        x2 = int(c.number_input("X2 · px", min_value=0, max_value=max(0, width-1), value=x2_default, step=1, key=key+"_x2"))
        y2 = int(d.number_input("Y2 · px", min_value=0, max_value=max(0, height-1), value=y2_default, step=1, key=key+"_y2"))
    elif mode == "文字":
        label = c.text_input("标注文字", max_chars=120, key=key+"_text")
        d.caption("文字位置由 X1 / Y1 决定")

    actions = st.columns(4, gap="small")
    add_clicked = actions[0].button("添加标注", type="primary", key=key+"_add", width="stretch",
                                     disabled=(mode == "文字" and not label.strip()))
    undo_clicked = actions[1].button("撤销", key=key+"_undo", width="stretch", disabled=not history["undo"])
    redo_clicked = actions[2].button("重做", key=key+"_redo", width="stretch", disabled=not history["redo"])
    clear_clicked = actions[3].button("清空全部", key=key+"_clear", width="stretch", disabled=not marks)

    def commit(next_marks):
        history["undo"].append(snapshot(marks))
        history["redo"].clear()
        history["last"] = snapshot(next_marks)
        return validate_annotations(next_marks, image.size)

    if add_clicked:
        points = [[x1, y1]] if mode in {"点位", "文字"} else [[x1, y1], [x2, y2]]
        kind = {"点位": "point", "距离测量": "line", "矩形 ROI": "rect", "文字": "text"}[mode]
        if kind in {"line", "rect"} and points[0] == points[1]:
            st.warning("请为线段或 ROI 选择两个不同的坐标点。")
            return None
        item = {"id": uuid.uuid4().hex[:12], "type": kind, "points": points}
        if kind == "text":
            item["text"] = label.strip()
        return commit(marks + [item])
    if undo_clicked:
        previous = history["undo"].pop()
        history["redo"].append(snapshot(marks))
        history["last"] = snapshot(previous)
        return previous
    if redo_clicked:
        following = history["redo"].pop()
        history["undo"].append(snapshot(marks))
        history["last"] = snapshot(following)
        return following
    if clear_clicked:
        return commit([])
    if marks:
        st.caption(f"当前有 {len(marks)} 项标注；修改会写入本机案例库，并包含在导出报告中。")
    return None


def comparison_canvas(image_a: Image.Image, image_b: Image.Image,
                      label_a: str, label_b: str, key: str, linked=True):
    """Read-only HTML viewer with linked wheel zoom and pointer pan."""
    src_a, src_b = image_data_uri(image_a), image_data_uri(image_b)
    title_a, title_b = escape(label_a), escape(label_b)
    checked = "checked" if linked else ""
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>
    *{{box-sizing:border-box}}body{{margin:0;font:12px/1.45 'Microsoft YaHei',sans-serif;color:#334155;background:#e9edf3}}
    .tools{{height:34px;display:flex;align-items:center;gap:14px}}button{{border:1px solid #ffffffb8;background:#e9edf3;border-radius:10px;padding:6px 11px;color:#334155;box-shadow:3px 3px 7px #c4cbd5,-3px -3px 7px #ffffff;cursor:pointer}}
    button:active{{box-shadow:inset 2px 2px 5px #c4cbd5,inset -2px -2px 5px #ffffff}}
    label{{display:flex;align-items:center;gap:6px}}.hint{{margin-left:auto;color:#64748b;font-size:11px}}
    .pair{{height:550px;display:flex;gap:8px}}.panel{{position:relative;flex:1;min-width:0;overflow:hidden;background:#07111c;border:1px solid #253447;border-radius:15px;touch-action:none;cursor:grab}}
    .panel:active{{cursor:grabbing}}.panel img{{position:absolute;left:50%;top:50%;width:100%;height:100%;object-fit:contain;transform:translate(calc(-50% + var(--x,0px)),calc(-50% + var(--y,0px))) scale(var(--zoom,1));transform-origin:center;user-select:none;-webkit-user-drag:none}}
    .tag{{position:absolute;z-index:2;left:10px;top:10px;max-width:90%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#dbeafe;background:#0b1725d9;border-radius:4px;padding:5px 8px}}
    </style><body><div class="tools"><button id="reset" type="button">适应窗口</button><label><input id="linked" type="checkbox" {checked}>同步缩放和平移</label><span class="hint">滚轮缩放 · 拖动平移</span></div>
    <div class="pair"><div class="panel" data-index="0"><span class="tag">{title_a}</span><img draggable="false" src="{src_a}" alt="A 侧影像"></div>
    <div class="panel" data-index="1"><span class="tag">{title_b}</span><img draggable="false" src="{src_b}" alt="B 侧影像"></div></div>
    <script>
    (()=>{{const panels=[...document.querySelectorAll('.panel')],linked=document.getElementById('linked');
    const views=[{{zoom:1,x:0,y:0}},{{zoom:1,x:0,y:0}}];
    function apply(i){{const img=panels[i].querySelector('img'),v=views[i];img.style.setProperty('--zoom',v.zoom);img.style.setProperty('--x',v.x+'px');img.style.setProperty('--y',v.y+'px');}}
    function targets(i){{return linked.checked?[0,1]:[i];}}
    panels.forEach((panel,i)=>{{let drag=null;
      panel.addEventListener('wheel',e=>{{e.preventDefault();const r=panel.getBoundingClientRect(),v=views[i],old=v.zoom,next=Math.max(1,Math.min(5,old*(e.deltaY<0?1.12:.89)));
        const px=e.clientX-r.left-r.width/2-v.x,py=e.clientY-r.top-r.height/2-v.y;
        targets(i).forEach(j=>{{const q=views[j];q.x+=px*(1-next/old);q.y+=py*(1-next/old);q.zoom=next;apply(j);}});
      }},{{passive:false}});
      panel.addEventListener('pointerdown',e=>{{drag={{x:e.clientX,y:e.clientY,views:views.map(v=>({{...v}}))}};panel.setPointerCapture(e.pointerId);}});
      panel.addEventListener('pointermove',e=>{{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;targets(i).forEach(j=>{{views[j].x=drag.views[j].x+dx;views[j].y=drag.views[j].y+dy;apply(j);}});}});
      panel.addEventListener('pointerup',()=>drag=null);panel.addEventListener('pointercancel',()=>drag=null);
    }});
    document.getElementById('reset').addEventListener('click',()=>{{views.forEach((v,i)=>{{views[i]={{zoom:1,x:0,y:0}};apply(i);}});}});
    }})();</script></body></html>"""
    st.iframe(html, width="stretch", height=600)
