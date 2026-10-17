"""Transactional local case archive. No image or patient data is sent remotely."""
import json
import os
import sqlite3
from io import BytesIO
from pathlib import Path
from PIL import Image
from app_support import ROOT, png_bytes

def connect():
    path = Path(os.environ.get("OCT_CASE_DB", str(ROOT / "outputs" / "cases.sqlite3")))
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS cases (
        id TEXT PRIMARY KEY, created TEXT NOT NULL, source TEXT NOT NULL,
        category TEXT NOT NULL, metadata TEXT NOT NULL, note TEXT NOT NULL,
        original BLOB NOT NULL, overlay BLOB)""")
    return con

def save_case(record):
    meta = {k:v for k,v in record.items() if k not in {"original","overlay","heatmap","note"}}
    with connect() as con:
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
    with connect() as con:
        rows = [dict(r) for r in con.execute(sql+" ORDER BY created DESC,id DESC",args)]
    con.close()
    return rows

def load_case(case_id):
    with connect() as con:
        row = con.execute("SELECT * FROM cases WHERE id=?",(case_id,)).fetchone()
    con.close()
    if row is None:
        return None
    record = json.loads(row["metadata"])
    record["note"] = row["note"]
    record["heatmap"] = None
    for key in ("original","overlay"):
        record[key] = Image.open(BytesIO(row[key])).convert("RGB") if row[key] else None
    return record

def update_note(case_id,note):
    with connect() as con:
        con.execute("UPDATE cases SET note=? WHERE id=?",(note,case_id))
    con.close()

def delete_case(case_id):
    with connect() as con:
        con.execute("DELETE FROM cases WHERE id=?",(case_id,))
    con.close()
