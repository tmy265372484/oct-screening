import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from case_library import save_case,load_case,list_cases,update_note,delete_case
from health_guide import reply, generate_answer

class CaseArchiveTests(unittest.TestCase):
    def test_roundtrip_search_update_and_delete_across_connections(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{"OCT_CASE_DB":str(Path(tmp)/"test.sqlite")}):
            record=dict(case_id="test-1",created_at="2026-10-18",source_name="example.png",
                        source_sha256="abc",note="",prediction={"predicted_class":"CNV"},
                        original=Image.new("RGB",(8,6),"gray"),overlay=Image.new("RGB",(8,6),"red"),
                        heatmap=None)
            save_case(record)
            loaded=load_case("test-1")
            self.assertEqual(loaded["original"].size,(8,6))
            self.assertEqual(loaded["overlay"].getpixel((0,0)),(255,0,0))
            update_note("test-1","视物变形")
            self.assertEqual(list_cases("变形","CNV")[0]["id"],"test-1")
            self.assertEqual(list_cases("' OR 1=1 --"),[])
            self.assertEqual(load_case("test-1")["note"],"视物变形")
            delete_case("test-1")
            self.assertIsNone(load_case("test-1"))
            self.assertEqual(list_cases(),[])

class HealthGuideTests(unittest.TestCase):
    def test_red_flag_precedes_normal_class_and_treatment_request(self):
        self.assertIn("急诊",reply("突然看不见，吃什么药",category="NORMAL"))
    def test_treatment_is_general_not_prescription(self):
        self.assertIn("单靠模型分类不能决定方案",reply("治疗方案",category="CNV"))
        self.assertIn("先别自行改药",reply("治疗方案",category="CNV"))
    def test_unknown_question_discloses_local_rule_limit(self):
        self.assertIn("本地规则参考",generate_answer("请帮我检查眼睛",[],None,""))
