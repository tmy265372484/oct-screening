import tempfile
from pathlib import Path
import unittest
from PIL import Image
from tools.audit_oct import inspect_image, summarize


class DataAuditTests(unittest.TestCase):
    def test_reencoded_pixels_and_candidate_ids_are_distinct_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root/'train/DRUSEN/DRUSEN-123-1.png',
                     root/'val/DRUSEN/DRUSEN-123-2.png',
                     root/'test/DRUSEN/DRUSEN-999-1.png']
            for i, p in enumerate(paths):
                p.parent.mkdir(parents=True)
                Image.new('L', (16, 16), 70 if i < 2 else 140).save(p, compress_level=i)
            report = summarize([inspect_image(p, root) for p in paths])
            pair = report['pairs']['train_val']
            self.assertEqual(pair['candidate_id']['shared_values'], 1)
            self.assertEqual(pair['file_sha256']['shared_values'], 0)
            self.assertEqual(pair['pixel_sha256']['shared_values'], 1)
            self.assertEqual(report['pairs']['train_test']['pixel_sha256']['shared_values'], 0)

    def test_corruption_is_recorded_and_unparsed_id_not_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root/'train/CNV/unknown.jpeg'
            p.parent.mkdir(parents=True)
            p.write_bytes(b'broken')
            report = summarize([inspect_image(p, root)])
            self.assertEqual(report['decode_errors'], 1)
            self.assertEqual(report['unparsed_filenames']['train'], 1)
            self.assertEqual(report['candidate_id_counts']['train'], 0)


if __name__ == '__main__':
    unittest.main()
