"""Fail-closed deployment checks; never print secret values."""
import importlib.util
import os
from pathlib import Path
import tempfile
import tomllib
from urllib.parse import urlparse


def validate_config(config):
    auth, access = config.get('auth', {}), config.get('access', {})
    for name in ('redirect_uri','cookie_secret','client_id','client_secret','server_metadata_url'):
        value = auth.get(name)
        if not isinstance(value,str) or not value.strip() or 'REPLACE' in value or 'YOUR-' in value:
            raise ValueError('OIDC configuration incomplete: '+name)
    for name in ('redirect_uri','server_metadata_url'):
        if urlparse(auth[name]).scheme != 'https' or not urlparse(auth[name]).netloc:
            raise ValueError('Production OIDC URLs must use HTTPS')
    if not auth['redirect_uri'].endswith('/oauth2callback'):
        raise ValueError('OIDC callback must end in /oauth2callback')
    if len(auth['cookie_secret']) < 32:
        raise ValueError('Cookie signing secret is too short')
    subjects=access.get('allowed_subjects')
    if not isinstance(subjects,list) or not subjects or any(not isinstance(s,str) or not s.strip() or 'REPLACE' in s for s in subjects):
        raise ValueError('Configure approved account subjects')
    issuer=access.get('issuer','')
    if not isinstance(issuer,str) or urlparse(issuer).scheme != 'https' or 'YOUR-' in issuer:
        raise ValueError('Configure expected OIDC issuer')


def main():
    if os.getenv('OCT_DEPLOYED') != '1' or os.getenv('OCT_AUTH_MODE') != 'oidc':
        raise ValueError('Production requires OIDC; anonymous fallback is disabled')
    validate_config(tomllib.loads(Path('.streamlit/secrets.toml').read_text(encoding='utf-8')))
    if importlib.util.find_spec('authlib') is None:
        raise ValueError('Install streamlit[auth]')
    model=Path(os.environ['OCT_MODEL_PATH'])
    if not model.is_file() or model.stat().st_size < 1024*1024:
        raise ValueError('Mount the trusted trained best.pt file')
    directory=Path(os.environ['OCT_DATA_DIR']); directory.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryFile(dir=directory) as test:
        test.write(b'write-check')
    print('Deployment checks passed. DeepSeek availability is independent of startup.')

if __name__=='__main__':
    try: main()
    except Exception as error:
        # Do not emit parser contents, configuration values or exception dumps.
        print('Deployment check failed ('+type(error).__name__+'). Check OIDC configuration, model mount and data permissions.')
        raise SystemExit(1)
