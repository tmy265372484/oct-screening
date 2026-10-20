import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import streamlit as st
from streamlit.testing.v1 import AppTest
from local_accounts import register, authenticate, resolve_session, revoke, connection
from access_control import database_path


class LocalAccountsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'OCT_AUTH_MODE':'accounts', 'OCT_DEPLOYED':'0',
            'OCT_AUTH_DB':str(Path(self.tmp.name)/'auth.sqlite'), 'OCT_DATA_DIR':self.tmp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop(); self.tmp.cleanup()

    def test_password_session_and_account_boundary(self):
        register('alice', 'good-password-123'); register('bob', 'good-password-456')
        with self.assertRaises(ValueError): authenticate('alice', 'wrong')
        a=authenticate('alice','good-password-123'); b=authenticate('bob','good-password-456')
        with patch.object(st,'session_state',{'_login_token':a}): pa=database_path()
        with patch.object(st,'session_state',{'_login_token':b}): pb=database_path()
        self.assertNotEqual(pa,pb)
        with self.assertRaises(PermissionError): resolve_session('forged')
        with connection() as con:
            digest=con.execute('SELECT digest FROM users WHERE name=?',('alice',)).fetchone()[0]
            self.assertNotEqual(digest, b'good-password-123')
        revoke(a)
        with self.assertRaises(PermissionError): resolve_session(a)
        self.assertTrue(resolve_session(b))

    def test_lockout_and_expiry(self):
        register('alice','good-password-123')
        for i in range(5):
            with self.assertRaises(ValueError): authenticate('alice','wrong')
        with self.assertRaisesRegex(ValueError,'5 分钟'): authenticate('alice','good-password-123')
        with connection() as con: con.execute('UPDATE users SET blocked=0,failures=0')
        token=authenticate('alice','good-password-123')
        with connection() as con: con.execute('UPDATE sessions SET expires=0')
        with self.assertRaises(PermissionError): resolve_session(token)

    def test_login_ui_register_login_logout(self):
        app=AppTest.from_string('from access_control import require_access, render_logout\nimport streamlit as st\nrequire_access()\nst.success("PRIVATE")\nrender_logout()').run()
        self.assertFalse(app.exception)
        app.text_input(key='signup_name').set_value('alice')
        app.text_input(key='signup_password').set_value('good-password-123')
        app.text_input(key='signup_confirm').set_value('good-password-123')
        next(b for b in app.button if b.label=='创建账号').click().run()
        self.assertFalse(app.exception)
        app.text_input(key='login_name').set_value('alice')
        app.text_input(key='login_password').set_value('good-password-123')
        next(b for b in app.button if b.label=='登录工作站').click().run()
        self.assertFalse(app.exception)
        self.assertIn('PRIVATE',[s.value for s in app.success])
        app.button(key='account_logout').click().run()
        self.assertFalse(app.exception)
        self.assertNotIn('PRIVATE',[s.value for s in app.success])

if __name__=='__main__': unittest.main()
