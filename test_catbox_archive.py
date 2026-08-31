import os
import sys
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch


mock_runpod = types.ModuleType("runpod")
mock_runpod.serverless = types.ModuleType("serverless")
mock_runpod.serverless.start = lambda config: None
mock_runpod.serverless.utils = types.ModuleType("utils")
mock_runpod.serverless.utils.rp_upload = lambda value: None
sys.modules.setdefault("runpod", mock_runpod)
sys.modules.setdefault("runpod.serverless", mock_runpod.serverless)
sys.modules.setdefault("runpod.serverless.utils", mock_runpod.serverless.utils)

import handler


class FakeResponse:
    def __init__(self, body=b""):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


class CatboxUploadTests(unittest.TestCase):
    def test_base64_data_uri_is_decoded_before_upload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = handler.save_base64_to_file(
                "data:image/png;base64,ZXhhY3QtaW1hZ2U=",
                temp_dir,
                "input.png",
            )
            self.assertEqual(b"exact-image", Path(image_path).read_bytes())

    def test_uploads_exact_file_with_hash_and_returns_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.jpg"
            image_path.write_bytes(b"exact-image-bytes")
            calls = []

            def fake_urlopen(request, timeout):
                calls.append((request, timeout))
                return FakeResponse(b"https://files.catbox.moe/abc123.jpg\n")

            with patch.object(handler.urllib.request, "urlopen", side_effect=fake_urlopen):
                saved_url = handler.upload_file_to_catbox(str(image_path), "secret-hash")

            self.assertEqual("https://files.catbox.moe/abc123.jpg", saved_url)
            self.assertEqual(1, len(calls))
            request, timeout = calls[0]
            self.assertEqual(handler.CATBOX_API_URL, request.full_url)
            self.assertEqual(handler.CATBOX_UPLOAD_TIMEOUT_SECONDS, timeout)
            self.assertIn(b"exact-image-bytes", request.data)
            self.assertIn(b"secret-hash", request.data)
            self.assertIn(b'name="fileToUpload"', request.data)

    def test_invalid_response_retries_then_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.jpg"
            image_path.write_bytes(b"image")
            with (
                patch.object(
                    handler.urllib.request,
                    "urlopen",
                    return_value=FakeResponse(b"upload failed"),
                ) as urlopen,
                patch.object(handler.time, "sleep") as sleep,
            ):
                with self.assertRaisesRegex(Exception, "Catbox upload failed"):
                    handler.upload_file_to_catbox(str(image_path), "secret-hash")

            self.assertEqual(handler.CATBOX_UPLOAD_ATTEMPTS, urlopen.call_count)
            sleep.assert_called_once_with(handler.CATBOX_RETRY_DELAY_SECONDS)

    def test_missing_hash_is_a_nonfatal_warning(self):
        saved_url, warning = handler.archive_resolved_input(
            "input.jpg", "base64", "", "start image"
        )
        self.assertIsNone(saved_url)
        self.assertEqual("start image: catbox_userhash is missing", warning)

    def test_path_input_is_not_uploaded(self):
        with patch.object(handler, "upload_file_to_catbox") as upload:
            saved_url, warning = handler.archive_resolved_input(
                "/input.jpg", "path", "secret-hash", "start image"
            )
        self.assertIsNone(saved_url)
        self.assertIsNone(warning)
        upload.assert_not_called()

    def test_job_input_redaction_hides_hash_and_base64(self):
        safe = handler.redacted_job_input({
            "catbox_userhash": "secret-hash",
            "image_base64": "aGVsbG8=",
            "end_image": "d29ybGQ=",
            "prompt": "hello",
        })
        rendered = repr(safe)
        self.assertNotIn("secret-hash", rendered)
        self.assertNotIn("aGVsbG8=", rendered)
        self.assertNotIn("d29ybGQ=", rendered)
        self.assertEqual("hello", safe["prompt"])


class HandlerArchiveErrorTests(unittest.TestCase):
    def test_bad_end_image_still_returns_start_archive_url(self):
        def fake_process(input_data, temp_dir, output_filename, input_type):
            if output_filename == "end_image.jpg":
                raise Exception("bad end image")
            return "/tmp/start.jpg"

        with (
            patch.object(handler, "reclaim_worker_disk"),
            patch.object(handler, "cleanup_job_files"),
            patch.object(handler, "process_input", side_effect=fake_process),
            patch.object(
                handler,
                "archive_resolved_input",
                return_value=("https://files.catbox.moe/start.jpg", None),
            ) as archive,
        ):
            result = handler.handler({
                "input": {
                    "image_url": "https://example.com/start.jpg",
                    "end_image_url": "https://example.com/end.jpg",
                    "catbox_userhash": "secret-hash",
                    "prompt": "hello",
                }
            })

        self.assertEqual("bad end image", result["error"])
        self.assertEqual("https://files.catbox.moe/start.jpg", result["saved_input_url"])
        self.assertIsNone(result["saved_end_input_url"])
        self.assertEqual([], result["input_upload_warnings"])
        archive.assert_called_once()

    def test_later_exception_includes_archive_fields(self):
        with (
            patch.object(handler, "reclaim_worker_disk"),
            patch.object(handler, "cleanup_job_files"),
            patch.object(handler, "process_input", return_value="/tmp/start.jpg"),
            patch.object(
                handler,
                "archive_resolved_input",
                return_value=("https://files.catbox.moe/start.jpg", None),
            ),
            patch.object(handler, "load_workflow", side_effect=Exception("Comfy boom")),
        ):
            result = handler.handler({
                "input": {
                    "image_url": "https://example.com/start.jpg",
                    "catbox_userhash": "secret-hash",
                    "prompt": "hello",
                }
            })

        self.assertEqual("Comfy boom", result["error"])
        self.assertEqual("https://files.catbox.moe/start.jpg", result["saved_input_url"])
        self.assertIsNone(result["saved_end_input_url"])
        self.assertEqual([], result["input_upload_warnings"])
        self.assertNotIn("video", result)


class DiskCleanupTests(unittest.TestCase):
    def test_reclaim_worker_disk_removes_leftover_videos_and_task_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            input_dir = Path(temp_dir) / "input"
            comfy_temp = Path(temp_dir) / "temp"
            output_dir.mkdir()
            input_dir.mkdir()
            comfy_temp.mkdir()
            leftover = output_dir / "25-12-03"
            leftover.mkdir()
            (leftover / "old.mp4").write_bytes(b"video")
            (input_dir / "task_old_input_image.jpg").write_bytes(b"img")
            (input_dir / "example_image.png").write_bytes(b"keep")
            (comfy_temp / "preview.png").write_bytes(b"tmp")

            with patch.dict(os.environ, {
                "COMFY_OUTPUT_DIR": str(output_dir),
                "COMFY_INPUT_DIR": str(input_dir),
                "COMFY_TEMP_DIR": str(comfy_temp),
            }):
                handler.reclaim_worker_disk()

            self.assertFalse(leftover.exists())
            self.assertFalse((input_dir / "task_old_input_image.jpg").exists())
            self.assertFalse((comfy_temp / "preview.png").exists())
            self.assertEqual(b"keep", (input_dir / "example_image.png").read_bytes())

    def test_get_videos_deletes_output_file_after_read(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "task_1_00001.mp4"
            video_path.write_bytes(b"mp4-bytes")
            ws = Mock()
            ws.recv.return_value = '{"type":"executing","data":{"node":null,"prompt_id":"prompt-1"}}'
            history = {
                "prompt-1": {
                    "status": {"status_str": "success"},
                    "outputs": {
                        "277": {
                            "gifs": [{"fullpath": str(video_path)}],
                        }
                    },
                }
            }
            with (
                patch.object(handler, "queue_prompt", return_value={"prompt_id": "prompt-1"}),
                patch.object(handler, "get_history", return_value=history),
            ):
                videos = handler.get_videos(ws, {})

            encoded = handler.base64.b64encode(b"mp4-bytes").decode("utf-8")
            self.assertEqual({"277": [encoded]}, videos)
            self.assertFalse(video_path.exists())

    def test_set_job_video_prefix_uses_task_id_without_dated_folders(self):
        prompt = {
            "277": {
                "class_type": "VHS_VideoCombine",
                "inputs": {"filename_prefix": "25-12-03/wan/25-12-03-FASTWAN"},
            }
        }
        handler.set_job_video_prefix(prompt, "task_abc")
        self.assertEqual("task_abc", prompt["277"]["inputs"]["filename_prefix"])


class ComfyRetryTests(unittest.TestCase):
    def test_http_connection_refusal_retries_without_time_shadowing(self):
        refused = urllib.error.URLError(ConnectionRefusedError(111, "refused"))
        with (
            patch.object(
                handler.urllib.request,
                "urlopen",
                side_effect=[refused, FakeResponse()],
            ) as urlopen,
            patch.object(handler.time, "sleep") as sleep,
        ):
            handler.wait_for_http_server(
                "http://127.0.0.1:8188/", max_attempts=2, retry_delay=1
            )

        self.assertEqual(2, urlopen.call_count)
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
