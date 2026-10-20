import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import requests
from PIL import Image
from case_library import save_case, load_case, save_review, review_history, delete_case
from health_guide import case_context, generate_answer
from reporting import build_report_html, build_report_markdown
from tools.prepare_grouped_split import prepare


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        env=patch.dict(os.environ,{'OCT_CASE_DB':str(Path(self.tmp.name)/'cases.sqlite')})
        env.start(); self.addCleanup(env.stop)
        self.record=dict(case_id='sample', created_at='2026-10-18', source_name='private.png',
                         source_sha256='abc',note='private note',original=Image.new('RGB',(4,4)),
                         prediction={'predicted_class':'CNV','probabilities':{'CNV':.8,'DME':.1,'DRUSEN':.05,'NORMAL':.05}})
        save_case(self.record)

    def test_review_roundtrip_keeps_model_and_rejects_stale_revision(self):
        save_review('sample','reader-1','修正分类','DME','<script>reason</script>',0)
        loaded=load_case('sample')
        self.assertEqual(loaded['prediction']['predicted_class'],'CNV')
        self.assertEqual(loaded['reviews'][0]['category'],'DME')
        save_case(self.record) # Stale sessions cannot erase review history.
        self.assertEqual(len(review_history('sample')),1)
        with self.assertRaises(ValueError): save_review('sample','reader-2','暂不能判断',None,'more scans',0)
        save_review('sample','reader-2','暂不能判断',None,'more scans',1)
        self.assertIsNone(load_case('sample')['reviews'][-1]['category'])
        html=build_report_html(load_case('sample'))
        self.assertIn('&lt;script&gt;',html); self.assertNotIn('<script>',html)
        self.assertIn('未电子签署',html)
        self.assertIn('reader',build_report_markdown(load_case('sample')))
        delete_case('sample'); self.assertEqual(review_history('sample'),[])

    def test_context_allowlist_and_consent(self):
        context=case_context(self.record)
        self.assertNotIn('private',str(context)); self.assertNotIn('abc',str(context))
        with patch('health_guide.ask_deepseek',return_value='reference') as call:
            generate_answer('explain',[],None,'test-key',context,False)
            self.assertNotIn('model_class',str(call.call_args))
            generate_answer('explain',[],None,'test-key',context,True)
            self.assertIn('model_class',str(call.call_args))

    def test_error_reason_does_not_expose_exception_or_key(self):
        with patch('health_guide.ask_deepseek',side_effect=requests.Timeout('private-key')):
            text=generate_answer('explain',[],None,'private-key')
        self.assertIn('响应超时',text); self.assertNotIn('private-key',text)

    def test_review_ui_form(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_string("from case_library import load_case\nfrom clinical_review import review_panel\nreview_panel(load_case('sample'),'test')",default_timeout=20).run()
        self.assertFalse(app.exception)
        app.text_input[0].set_value('reader-ui')
        app.text_area[0].set_value('Requires complete OCT series')
        app.checkbox[0].check()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(review_history('sample')),1)


class GroupedSplitTests(unittest.TestCase):
    def test_group_transitivity_test_lock_and_duplicate_removal(self):
        def r(path,split,patient,pixels,category='CNV'):
            return dict(path=path,split=split,candidate_id=patient,pixel_sha256=pixels,file_sha256=pixels,**{'class':category})
        rows=[r('a','train','1','x'),r('b','val','2','x'),r('c','train','2','y'),
              r('d','test','3','z'),r('e','train','3','w'),r('f','train','4','bad','CNV'),r('g','val','5','bad','DME')]
        output=prepare(rows)
        self.assertEqual([r['path'] for r in output['test']],['d'])
        kept=output['train'] or output['val']
        self.assertEqual({r['path'] for r in kept},{'a','c'})
        self.assertEqual({r['path'] for r in output['quarantine']},{'b','e','f','g'})

if __name__=='__main__': unittest.main()
