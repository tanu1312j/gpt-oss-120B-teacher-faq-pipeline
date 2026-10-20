import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_teacher import generate, quality_flags, read_faq
from export_reviewed import export


class PipelineTests(unittest.TestCase):
    def test_numeric_quality_check(self):
        self.assertEqual(quality_flags("Within 7 days", "Within 7 business days", "stop"), [])
        self.assertIn("reference_numbers_missing:7", quality_flags("Within 7 days", "Soon", "stop"))
        self.assertIn("output_truncated", quality_flags("No refund", "No refund", "length"))

    def test_end_to_end_mocked_teacher_and_manual_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_file = root / "faq.csv"
            csv_file.write_text(
                "id,question,reference_answer,category\nA,What is the limit?,The limit is 7 days.,support\n",
                encoding="utf-8"
            )
            output = root / "teacher.jsonl"
            with patch("generate_teacher.get_answer", return_value=("The limit is 7 days.", "stop", {"total_tokens": 42})) as fake:
                made, skipped = generate(csv_file, output, "http://127.0.0.1:8080", "gpt-oss-120b", None, 10, 128)
                self.assertEqual((made, skipped), (1, 0))
                made, skipped = generate(csv_file, output, "http://127.0.0.1:8080", "gpt-oss-120b", None, 10, 128)
                self.assertEqual((made, skipped), (0, 1))
                self.assertEqual(fake.call_count, 1)
            saved = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(saved["review_status"], "pending_human_review")
            self.assertNotIn("reasoning_content", saved)
            approved = root / "approved.txt"
            approved.write_text("A\n", encoding="utf-8")
            target = root / "sft.jsonl"
            self.assertEqual(export(output, approved, target), 1)
            exported = json.loads(target.read_text(encoding="utf-8").strip())
            self.assertEqual(exported["messages"][1]["content"], "The limit is 7 days.")

    def test_invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("id,question,reference_answer\nA,Question,\n")
            with self.assertRaises(ValueError):
                read_faq(path)


if __name__ == "__main__":
    unittest.main()
