import json
import pytest
from vasts import json_output


def test_atomic_write_preserves_unicode_and_valid_json(tmp_path):
    path = tmp_path / "结果.json"
    json_output.write_json_atomic(path, {"label": "目标"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"label": "目标"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_failed_replace_and_serialization_preserve_previous_result(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text("previous", encoding="utf-8")
    def fail(*args): raise OSError("simulated replacement failure")
    monkeypatch.setattr(json_output.os, "replace", fail)
    with pytest.raises(OSError):
        json_output.write_json_atomic(path, {"new": 1})
    assert path.read_text() == "previous"
    assert list(tmp_path.glob("*.tmp")) == []
    with pytest.raises(ValueError):
        json_output.write_json_atomic(path, {"bad": float("nan")})
    assert path.read_text() == "previous"
