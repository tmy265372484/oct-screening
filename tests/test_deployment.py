import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from access_control import auth_mode, identity_key, database_path, verified_identity
from case_library import save_case, list_cases, load_case, update_note, delete_case, save_review, review_history
from deploy.preflight import validate_config


class DeploymentTests(unittest.TestCase):
    def test_deployed_never_uses_anonymous_fallback(self):
        with patch.dict(os.environ, {'OCT_DEPLOYED':'1','OCT_AUTH_MODE':'local'}):
            with self.assertRaises(PermissionError): auth_mode()
        with patch.dict(os.environ, {'OCT_AUTH_MODE':'oidc'}), patch('access_control.verified_identity',side_effect=PermissionError('login')):
            with self.assertRaises(PermissionError): database_path()

    def test_identity_and_allowlist(self):
        import streamlit as st
        class User(dict): is_logged_in=True
        with patch.object(st,'user',User(iss='https://id.test',sub='alice')), patch.object(st,'secrets',{'access':{'issuer':'https://id.test','allowed_subjects':['alice']}}):
            self.assertEqual(verified_identity(),identity_key('https://id.test','alice'))
        with patch.object(st,'user',User(iss='https://other.test',sub='alice')), patch.object(st,'secrets',{'access':{'issuer':'https://id.test','allowed_subjects':['alice']}}):
            with self.assertRaises(PermissionError): verified_identity()
        self.assertNotEqual(identity_key('one','alice'),identity_key('two','alice'))

    def test_all_case_operations_are_isolated_even_with_known_case_id(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'OCT_AUTH_MODE':'oidc','OCT_DATA_DIR':tmp,'OCT_CASE_DB':str(Path(tmp)/'shared.sqlite')}):
            a=identity_key('issuer','alice'); b=identity_key('issuer','bob')
            record=dict(case_id='known-id',created_at='today',source_name='image',note='',original=Image.new('RGB',(3,3)),prediction={'predicted_class':'CNV'})
            with patch('access_control.verified_identity',return_value=a):
                save_case(record)
                save_review('known-id','reader','确认模型分类',None,'reviewed',0)
            with patch('access_control.verified_identity',return_value=b):
                self.assertEqual(list_cases(),[]); self.assertIsNone(load_case('known-id'))
                self.assertEqual(review_history('known-id'),[])
                update_note('known-id','attack'); delete_case('known-id')
                with self.assertRaises(ValueError): save_review('known-id','bob','暂不能判断',None,'x',0)
                save_case(record) # Same ID in another account is not an overwrite.
            with patch('access_control.verified_identity',return_value=a):
                self.assertEqual(load_case('known-id')['note'],'')
                self.assertEqual(len(review_history('known-id')),1)
            self.assertFalse((Path(tmp)/'shared.sqlite').exists())

    def test_placeholder_configuration_rejected(self):
        import tomllib
        config=tomllib.loads(Path('deploy/secrets.example.toml').read_text(encoding='utf-8'))
        with self.assertRaises(ValueError): validate_config(config)
        config={'auth':dict(redirect_uri='https://app.test/oauth2callback',cookie_secret='x'*48,client_id='client',client_secret='secret',server_metadata_url='https://id.test/.well-known/openid-configuration'), 'access':dict(issuer='https://id.test',allowed_subjects=['alice'])}
        validate_config(config)
        config['access']['allowed_subjects']=[]
        with self.assertRaises(ValueError): validate_config(config)

if __name__=='__main__': unittest.main()
