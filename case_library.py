"""Transactional local case archive. No image or patient data is sent remotely."""
from contextlib import closing
import json
import os
import sqlite3
from io import BytesIO
from pathlib import Path
from PIL import Image
from app_support import ROOT, png_bytes
from access_control import database_path

def connect():
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS cases (
        id TEXT PRIMARY KEY, created TEXT NOT NULL, source TEXT NOT NULL,
        category TEXT NOT NULL, metadata TEXT NOT NULL, note TEXT NOT NULL,
        original BLOB NOT NULL, overlay BLOB)""")
    con.execute("""CREATE TABLE IF NOT EXISTS reviews (
        case_id TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL,
        PRIMARY KEY(case_id, revision))""")
    return con

def save_case(record):
    meta = {k:v for k,v in record.items() if k not in {"original","overlay","heatmap","note","reviews"}}
    with closing(connect()) as con, con:
        con.execute("INSERT INTO cases VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET metadata=excluded.metadata,note=excluded.note,overlay=excluded.overlay",
                    (record["case_id"],record["created_at"],record["source_name"],
                     record["prediction"]["predicted_class"],json.dumps(meta,ensure_ascii=False),
                     record.get("note",""),png_bytes(record["original"]),
                     png_bytes(record["overlay"]) if record.get("overlay") is not None else None))
    con.close()

def list_cases(query="", category="全部"):
    sql = "SELECT id,created,source,category,note FROM cases WHERE (instr(source,?)>0 OR instr(id,?)>0 OR instr(note,?)>0)"
    args = [query,query,query]
    if category != "全部":
        sql += " AND category=?"
        args.append(category)
    with closing(connect()) as con, con:
        rows = [dict(r) for r in con.execute(sql+" ORDER BY created DESC,id DESC",args)]
    con.close()
    return rows

def load_case(case_id):
    with closing(connect()) as con, con:
        row = con.execute("SELECT * FROM cases WHERE id=?",(case_id,)).fetchone()
    con.close()
    if row is None:
        return None
    record = json.loads(row["metadata"])
    record["note"] = row["note"]
    record["heatmap"] = None
    record["reviews"] = review_history(case_id)
    for key in ("original","overlay"):
        record[key] = Image.open(BytesIO(row[key])).convert("RGB") if row[key] else None
    return record

def update_note(case_id,note):
    with closing(connect()) as con, con:
        con.execute("UPDATE cases SET note=? WHERE id=?",(note,case_id))
    con.close()

def delete_case(case_id):
    with closing(connect()) as con, con:
        con.execute("DELETE FROM reviews WHERE case_id=?",(case_id,))
        con.execute("DELETE FROM cases WHERE id=?",(case_id,))
    con.close()


def review_history(case_id):
    with closing(connect()) as con, con:
        rows = con.execute("SELECT payload FROM reviews WHERE case_id=? ORDER BY revision", (case_id,)).fetchall()
    con.close()
    return [json.loads(row[0]) for row in rows]


def save_review(case_id, reviewer, decision, category, rationale, expected_revision):
    """Append a review without altering model predictions; reject stale edits."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    if decision not in ("确认模型分类", "修正分类", "暂不能判断"):
        raise ValueError("请选择有效的复核结论。")
    if not reviewer.strip() or not rationale.strip():
        raise ValueError("请填写复核者标识和判断依据。")
    if len(reviewer) > 80 or len(rationale) > 2000:
        raise ValueError("复核记录超过长度限制。")
    with closing(connect()) as con, con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT metadata FROM cases WHERE id=?", (case_id,)).fetchone()
        if row is None:
            raise ValueError("案例不存在，请先保存案例。")
        revision = con.execute("SELECT COALESCE(MAX(revision),0) FROM reviews WHERE case_id=?", (case_id,)).fetchone()[0]
        if revision != expected_revision:
            raise ValueError("已有新的复核记录，请刷新页面后重新复核。")
        record = json.loads(row[0])
        predicted = record["prediction"]["predicted_class"]
        if decision == "确认模型分类": category = predicted
        elif decision == "暂不能判断": category = None
        elif category not in ("CNV", "DME", "DRUSEN", "NORMAL", "其他/超出模型范围") or category == predicted:
            raise ValueError("修正分类应选择与模型不同的类别。")
        payload = dict(revision=revision+1, reviewer=reviewer.strip(), decision=decision,
                       category=category, rationale=rationale.strip(),
                       reviewed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                       source_sha256=record.get("source_sha256"),
                       model_prediction=record["prediction"],
                       identity_note="本机填写的复核标识，未经身份认证，不是电子签名")
        con.execute("INSERT INTO reviews VALUES (?,?,?)", (case_id, revision+1, json.dumps(payload, ensure_ascii=False)))
    con.close()
    return payload
