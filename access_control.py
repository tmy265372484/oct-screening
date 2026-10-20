"""OIDC identity gate and server-derived per-account storage boundaries."""
import hashlib
import json
import os
from pathlib import Path
from app_support import ROOT


def auth_mode():
    mode = os.getenv('OCT_AUTH_MODE', 'local')
    if mode not in ('local', 'oidc', 'accounts') or (os.getenv('OCT_DEPLOYED') == '1' and mode != 'oidc'):
        raise PermissionError('部署版必须启用 OIDC 登录。')
    return mode


def identity_key(issuer, subject):
    if not isinstance(issuer, str) or not issuer or not isinstance(subject, str) or not subject:
        raise PermissionError('身份提供方没有返回有效的 issuer/sub。')
    return hashlib.sha256(json.dumps([issuer, subject], separators=(',', ':')).encode()).hexdigest()


def verified_identity():
    """Never accept an identity from widgets, URLs or session-state parameters."""
    import streamlit as st
    if auth_mode() == 'accounts':
        from local_accounts import resolve_session
        return resolve_session(st.session_state.get('_login_token', ''))
    if not st.user.is_logged_in:
        raise PermissionError('请先登录。')
    issuer, subject = st.user.get('iss'), st.user.get('sub')
    key = identity_key(issuer, subject)
    access = st.secrets.get('access', {})
    if issuer != access.get('issuer') or subject not in access.get('allowed_subjects', []):
        raise PermissionError('此账号未获授权，请联系系统管理员。')
    return key


def database_path():
    if auth_mode() == 'local':
        return Path(os.getenv('OCT_CASE_DB', str(ROOT/'outputs/cases.sqlite3')))
    # Shared OCT_CASE_DB overrides are deliberately ignored in OIDC mode.
    owner = verified_identity()
    root = Path(os.getenv('OCT_DATA_DIR', str(ROOT/'outputs/accounts'))).resolve()
    return root/owner/'cases.sqlite3'


def require_access():
    import streamlit as st
    try:
        mode = auth_mode()
    except PermissionError as error:
        st.error(str(error)); st.stop()
    if mode == 'local':
        return
    if mode == 'accounts':
        from local_accounts import login_gate
        login_gate()
        return
    try:
        auth = st.secrets.get('auth', {})
        access = st.secrets.get('access', {})
        configured = all(auth.get(k) for k in ('redirect_uri','cookie_secret','client_id','client_secret','server_metadata_url'))
        configured = configured and bool(access.get('issuer')) and bool(access.get('allowed_subjects'))
    except Exception:
        configured = False
    if not configured:
        st.error('登录配置未完成，请联系系统管理员。'); st.stop()
    if not st.user.is_logged_in:
        st.title('眸析 · 医生工作站')
        st.caption('登录后进入你的影像与案例工作空间。')
        if st.button('登录工作站', type='primary'):
            st.login()
        st.stop()
    try:
        identity = verified_identity()
    except PermissionError as error:
        st.error(str(error))
        if st.button('退出此账号'): st.logout()
        st.stop()
    if st.session_state.get('_account_identity') != identity:
        st.session_state.clear()
        st.session_state['_account_identity'] = identity
def render_logout():
    import streamlit as st
    if auth_mode() == 'accounts':
        from local_accounts import logout_button
        logout_button()
        return
    if auth_mode() == 'oidc' and st.button('退出登录', key='account_logout'):
        st.session_state.clear()
        st.logout()
        st.stop()
