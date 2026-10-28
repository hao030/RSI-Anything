"""Checksum-verified asset installation, independent of training dependencies."""
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request

CATALOG = Path(__file__).with_name('catalogs') / 'code_repair.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def matches(path, record):
    return path.is_file() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256']


def safe_path(root, name):
    rel = PurePosixPath(name)
    if not name or rel.is_absolute() or '..' in rel.parts or '\\' in name or str(rel) != name:
        raise ValueError('Invalid asset path: ' + name)
    path = root / name
    for part in (path, *path.parents):
        if part == root.parent:
            break
        if part.is_symlink():
            raise ValueError('Asset destination contains a symlink: ' + name)
    return path


class ScopedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != 'https':
            raise ValueError('Asset redirect requires HTTPS')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected


def download(record, target):
    if matches(target, record):
        return
    url = record['url']
    headers = {'User-Agent': 'RSI-Anything-assets/1', 'Accept': 'application/octet-stream'}
    if os.environ.get('GH_TOKEN') and record.get('api_url'):
        url = record['api_url']
        if urllib.parse.urlparse(url).netloc != 'api.github.com':
            raise ValueError('Invalid authenticated GitHub asset URL')
        headers['Authorization'] = 'Bearer ' + os.environ['GH_TOKEN']
    elif os.environ.get('HF_TOKEN') and urllib.parse.urlparse(url).netloc == 'huggingface.co':
        headers['Authorization'] = 'Bearer ' + os.environ['HF_TOKEN']
    if urllib.parse.urlparse(url).scheme != 'https':
        raise ValueError('Asset downloads require HTTPS')
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.download-', dir=target.parent)
    partial = Path(name)
    try:
        opener = urllib.request.build_opener(ScopedRedirect())
        request = urllib.request.Request(url, headers=headers)
        with os.fdopen(fd, 'wb') as out, opener.open(request, timeout=120) as response:
            count = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > record['bytes']:
                    raise ValueError('Download exceeds recorded size: ' + target.name)
                out.write(block)
        if not matches(partial, record):
            raise ValueError('Download checksum mismatch: ' + target.name)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def extract(archive, destination, files):
    """Only regular, explicitly catalogued files can be extracted."""
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = member.name
            if name in seen or name not in files or not member.isfile():
                raise ValueError('Unexpected archive member: ' + name)
            record = files[name]
            if member.size != record['bytes']:
                raise ValueError('Archive member size mismatch: ' + name)
            target = safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('xb') as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
            if not matches(target, record):
                raise ValueError('Extracted file checksum mismatch: ' + name)
            seen.add(name)
    if seen != set(files):
        raise ValueError('Archive is missing catalogued files')


def selected(catalog, profile, models=True):
    settings = catalog['profiles'][profile]
    files = {}
    for group in settings['bundles']:
        files.update(catalog['bundles'][group]['files'])
    if models:
        for role in settings['models']:
            model = catalog['models'][role]
            for name, info in model['files'].items():
                files['models/' + role + '/' + name] = dict(info, source=model['repo_id'], revision=model['revision'])
    return files


def install(output, profile='evaluate', *, models=True, verify_only=False, catalog_path=CATALOG):
    catalog = json.loads(Path(catalog_path).read_text())
    output = Path(output).absolute()
    if output.is_symlink():
        raise ValueError('Use a real asset directory')
    output.mkdir(parents=True, exist_ok=True)
    expected = selected(catalog, profile, models)
    with safe_path(output, '.install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if verify_only:
            for name, record in expected.items():
                if not matches(safe_path(output, name), record):
                    raise ValueError('Asset integrity failure: ' + name)
            return {'task': 'code_repair', 'profile': profile, 'verified_files': len(expected), 'models': models}
        manifest_path = safe_path(output, 'manifest.json')
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if previous and (previous.get('release') != catalog['release'] or previous.get('experiment_sha256') != catalog['experiment_sha256']):
            raise ValueError('Asset directory belongs to another export; choose a new output directory')
        # Existing valid files are reusable; unrelated or corrupted files require an explicit new directory.
        for name, record in expected.items():
            path = safe_path(output, name)
            if path.exists() and not matches(path, record):
                raise ValueError('Existing asset differs from release; choose a new directory: ' + name)
        cache = safe_path(output, '.downloads')
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.install-', dir=output) as temp:
            stage = Path(temp)
            for group in catalog['profiles'][profile]['bundles']:
                bundle = catalog['bundles'][group]
                if all(matches(safe_path(output, n), r) for n, r in bundle['files'].items()):
                    continue
                archive = safe_path(cache, bundle['filename'])
                print('Downloading ' + bundle['filename'], flush=True)
                download(bundle, archive)
                extract(archive, stage, bundle['files'])
                for name in bundle['files']:
                    target = safe_path(output, name)
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        (stage / name).replace(target)
            if models:
                for role in catalog['profiles'][profile]['models']:
                    model = catalog['models'][role]
                    for name, record in model['files'].items():
                        target = safe_path(output, 'models/' + role + '/' + name)
                        if matches(target, record):
                            continue
                        print('Downloading ' + role + '/' + name, flush=True)
                        url = 'https://huggingface.co/' + model['repo_id'] + '/resolve/' + model['revision'] + '/' + urllib.parse.quote(name)
                        download(dict(record, url=url), target)
            merged = dict(previous.get('files', {}), **expected)
            # Verify the complete installed set before sealing, including earlier full-profile files.
            for name, record in merged.items():
                if not matches(safe_path(output, name), record):
                    raise ValueError('Asset integrity failure: ' + name)
            manifest = dict(version=1, release=catalog['release'], task='code_repair', files=merged,
                experiment_sha256=catalog['experiment_sha256'], recorded_public_inputs=True,
                hidden_execution_results_included=False)
            tmp = stage / 'manifest.json'
            tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
            tmp.replace(manifest_path)
    return {'task': 'code_repair', 'profile': profile, 'installed_files': len(merged), 'models': models, 'output': str(output)}
