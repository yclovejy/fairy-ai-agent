from __future__ import annotations

import os
import tempfile
from typing import Any

import torch
from transformers import pipeline


class WhisperService:
    def __init__(self) -> None:
        self.model_id = os.getenv("WHISPER_MODEL_ID", "").strip()
        self.language = os.getenv("WHISPER_LANGUAGE", "zh")
        self.device = 0 if torch.cuda.is_available() else -1
        self._pipeline = None
        self._pipeline_error = ""

    def available(self) -> bool:
        return bool(self.model_id)

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        if not self.model_id:
            raise RuntimeError("未配置 WHISPER_MODEL_ID，当前使用前端浏览器语音识别。")

        try:
            self._pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.model_id,
                device=self.device,
            )
            return self._pipeline
        except Exception as exc:
            self._pipeline_error = str(exc)
            raise RuntimeError(f"Whisper 模型加载失败: {exc}") from exc

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_bytes)
            temp_path = tmp_file.name

        try:
            recognizer = self._get_pipeline()
            result = recognizer(temp_path, generate_kwargs={"language": self.language})
            return {
                "text": (result.get("text", "") or "").strip(),
                "provider": f"whisper:{self.model_id}",
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


whisper_service = WhisperService()
