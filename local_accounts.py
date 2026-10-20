"""Local account authentication. Passwords are salted scrypt hashes; sessions expire."""
from contextlib import contextmanager
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from app_support import ROOT


@contextmanager
def connection():
    path = Path(os.getenv('OCT_AUTH_DB', str(ROOT/'outputs/auth/accounts.sqlite3')))
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=15)
    try:
        con.executescript('''
        CREATE TABLE IF NOT EXISTS users (
          name TEXT PRIMARY KEY, uid TEXT UNIQUE NOT NULL, salt BLOB NOT NULL,
          digest BLOB NOT NULL, failures INTEGER DEFAULT 0, blocked REAL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, uid TEXT NOT NULL, expires REAL NOT NULL);
        ''')
        with con:
            yield con
    finally:
        con.close()


def normalize(name):
    name = name.strip().lower()
    if not re.fullmatch(r'[a-z0-9_\-]{3,32}', name):
        raise ValueError('用户名需为 3–32 位英文字母、数字、下划线或短横线。')
    return name


def password_hash(password, salt):
    return hashlib.scrypt(password.encode('utf-8'), salt=salt, n=16384, r=8, p=1, dklen=32)


def register(name, password):
    name = normalize(name)
    if not 10 <= len(password) <= 128:
        raise ValueError('密码长度需为 10–128 位。')
    salt = secrets.token_bytes(16)
    digest = password_hash(password, salt)
    try:
        with connection() as con:
            con.execute('INSERT INTO users(name,uid,salt,digest) VALUES (?,?,?,?)',
                        (name, secrets.token_hex(24), salt, digest))
    except sqlite3.IntegrityError:
        raise ValueError('该用户名已被使用。') from None


def authenticate(name, password):
    name = normalize(name)
    if len(password) > 128:
        raise ValueError('用户名或密码错误。')
    token = None
    with connection() as con:
        con.execute('BEGIN IMMEDIATE')
        row = con.execute('SELECT uid,salt,digest,failures,blocked FROM users WHERE name=?', (name,)).fetchone()
        now = time.time()
        if row and row[4] > now:
            error = '尝试次数过多，请 5 分钟后再试。'
        elif row and hmac.compare_digest(password_hash(password, row[1]), row[2]):
            token = secrets.token_urlsafe(32)
            con.execute('UPDATE users SET failures=0,blocked=0 WHERE name=?', (name,))
            con.execute('DELETE FROM sessions WHERE expires<?', (now,))
            con.execute('INSERT INTO sessions VALUES (?,?,?)', (hashlib.sha256(token.encode()).hexdigest(), row[0], now+28800))
        else:
            if row:
                failures = row[3]+1 if not row[4] else 1
                con.execute('UPDATE users SET failures=?,blocked=? WHERE name=?',
                            (failures, now+300 if failures >= 5 else 0, name))
            else:
                password_hash(password, b'unknown-user-pad')
            error = '用户名或密码错误。'
    if token is None:
        raise ValueError(error)
    return token


def resolve_session(token):
    if not isinstance(token, str) or not token:
        raise PermissionError('请先登录。')
    with connection() as con:
        row = con.execute('SELECT s.uid FROM sessions s JOIN users u ON s.uid=u.uid WHERE s.token=? AND s.expires>?',
                          (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
    if row is None:
        raise PermissionError('登录已过期，请重新登录。')
    return row[0]


def revoke(token):
    with connection() as con:
        con.execute('DELETE FROM sessions WHERE token=?', (hashlib.sha256(token.encode()).hexdigest(),))


def login_gate():
    import streamlit as st
    try:
        owner = resolve_session(st.session_state.get('_login_token', ''))
        if st.session_state.get('_account_identity') != owner:
            token = st.session_state['_login_token']
            st.session_state.clear()
            st.session_state.update(_login_token=token, _account_identity=owner)
        return
    except PermissionError:
        if st.session_state.get("_login_token") or st.session_state.get("_account_identity"):
            st.session_state.clear()
    _, center, _ = st.columns([1, 1.2, 1])
    with center, st.container(border=True):
        st.title('眸析 · 登录工作站')
        st.caption('独立账号 · 私有病例空间 · OCT 辅助分析')
        login, signup = st.tabs(['登录', '注册账号'])
        with login, st.form('local_login', clear_on_submit=True):
            name = st.text_input('用户名', key='login_name')
            password = st.text_input('密码', type='password', key='login_password')
            if st.form_submit_button('登录工作站', type='primary', width='stretch'):
                try:
                    token = authenticate(name, password)
                    st.session_state.clear()
                    st.session_state['_login_token'] = token
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
        with signup, st.form('local_signup', clear_on_submit=True):
            name = st.text_input('用户名', key='signup_name', help='3–32 位英文字母、数字、下划线或短横线')
            password = st.text_input('密码', type='password', key='signup_password', help='至少 10 位')
            confirm = st.text_input('确认密码', type='password', key='signup_confirm')
            if st.form_submit_button('创建账号', width='stretch'):
                try:
                    if password != confirm:
                        raise ValueError('两次密码输入不一致。')
                    register(name, password)
                    st.success('账号已创建，请切换到“登录”。')
                except ValueError as error:
                    st.error(str(error))
        st.caption('本机账号系统；新账号拥有空白病例库，原有病例文件保留。')
    st.stop()


def logout_button():
    import streamlit as st
    if st.button('退出登录', key='account_logout'):
        revoke(st.session_state.get('_login_token', ''))
        st.session_state.clear()
        st.rerun()
