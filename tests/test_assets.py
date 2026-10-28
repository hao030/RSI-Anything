import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rsi_anything import assets


def record(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


class AssetContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def archive(self, name, members):
        path = self.root / name
        with tarfile.open(path, 'w:gz') as tar:
            for member, content, kind in members:
                info = tarfile.TarInfo(member)
                info.type = kind
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
        return path

    def test_rejects_traversal_links_duplicates_and_missing_files(self):
        scenarios = [
            ([('../escape', b'a', tarfile.REGTYPE)], {'../escape': record(b'a')}),
            ([('file', b'', tarfile.SYMTYPE)], {'file': record(b'')}),
            ([('file', b'a', tarfile.REGTYPE)] * 2, {'file': record(b'a')}),
            ([], {'file': record(b'a')}),
            ([('file', b'b', tarfile.REGTYPE)], {'file': record(b'a')}),
        ]
        for i, (members, files) in enumerate(scenarios):
            with self.subTest(i=i), self.assertRaises(ValueError):
                assets.extract(self.archive(str(i) + '.tgz', members), self.root / str(i), files)
        self.assertFalse((self.root / 'escape').exists())

    def test_destination_symlink_rejected(self):
        folder = self.root / 'output'
        folder.mkdir()
        (folder / 'training').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            assets.safe_path(folder, 'training/file')

    def test_profile_upgrade_reuses_verified_files_and_retains_full_manifest(self):
        bundles = {}
        archives = {}
        for group, name, value in [('evaluation', 'public/a', b'public'), ('training', 'training/b', b'train')]:
            path = self.archive(group + '.tgz', [(name, value, tarfile.REGTYPE)])
            bundles[group] = dict(record(path.read_bytes()), filename=path.name, files={name: record(value)}, url='https://example.invalid/' + path.name)
            archives[path.name] = path
        catalog = {'release': 'test', 'experiment_sha256': 'test', 'bundles': bundles,
                   'models': {}, 'profiles': {'evaluate': {'bundles': ['evaluation'], 'models': []},
                                             'full': {'bundles': ['evaluation','training'], 'models': []}}}
        cp = self.root / 'catalog.json'
        cp.write_text(json.dumps(catalog))
        output = self.root / 'installed'
        def fake_download(info, path):
            path.write_bytes(archives[path.name].read_bytes())
        with patch.object(assets, 'download', side_effect=fake_download) as download:
            assets.install(output, 'evaluate', catalog_path=cp)
            assets.install(output, 'full', catalog_path=cp)
            result = assets.install(output, 'evaluate', catalog_path=cp)
            self.assertEqual(download.call_count, 2)
        self.assertEqual(result['installed_files'], 2)
        self.assertEqual(assets.install(output, 'full', verify_only=True, catalog_path=cp)['verified_files'], 2)
        (output / 'training/b').write_bytes(b'other')
        with self.assertRaises(ValueError):
            assets.install(output, 'full', verify_only=True, catalog_path=cp)

    def test_download_corruption_leaves_no_partial_target(self):
        class Response(io.BytesIO):
            pass
        class Opener:
            def open(self, request, timeout):
                return Response(b'bad')
        target = self.root / 'download'
        with patch('urllib.request.build_opener', return_value=Opener()):
            with self.assertRaises(ValueError):
                assets.download(dict(record(b'yes'), url='https://example.invalid/test'), target)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_auth_removed_on_cross_host_redirect(self):
        req = assets.urllib.request.Request('https://api.github.com/asset', headers={'Authorization': 'Bearer test'})
        redirected = assets.ScopedRedirect().redirect_request(req, None, 302, '', {}, 'https://objects.githubusercontent.com/blob')
        self.assertIsNone(redirected.get_header('Authorization'))
        with self.assertRaises(ValueError):
            assets.ScopedRedirect().redirect_request(req, None, 302, '', {}, 'http://example.invalid/blob')


if __name__ == '__main__':
    unittest.main()
