import os
import time
import zipfile

import pytest
import requests

from zandmotor import model_bundle
from zandmotor.config import CFG
from zandmotor.errors import ModelBundleError


def _fake_model(data, mtime=1_700_000_000):
    for i, f in enumerate(model_bundle.model_files()):
        p = data / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"file {i}".encode())
        os.utime(p, (mtime, mtime))


def _export(tmp_path):
    src = tmp_path / "maintainer"
    _fake_model(src)
    out = tmp_path / "release" / "zandmotor-model.zip"
    manifest = model_bundle.export_bundle(out, data=src)
    return out, manifest


def test_export_then_install_round_trip_keeps_mtimes(tmp_path):
    zip_path, manifest = _export(tmp_path)
    assert set(manifest["files"]) == set(model_bundle.model_files())
    assert (zip_path.parent / "manifest.json").exists()
    user = tmp_path / "user"
    model_bundle.install_bundle(zip_path.read_bytes(), data=user)
    for f in model_bundle.model_files():
        assert (user / f).read_bytes() == (tmp_path / "maintainer" / f).read_bytes()
        assert abs((user / f).stat().st_mtime - 1_700_000_000) < 2
    assert model_bundle.installed_manifest(user)["created"] == manifest["created"]


def test_export_refuses_an_incomplete_model(tmp_path):
    src = tmp_path / "m"
    _fake_model(src)
    (src / CFG["lagoon_basin_file"]).unlink()
    with pytest.raises(ModelBundleError, match="lagoon_basin"):
        model_bundle.export_bundle(tmp_path / "x.zip", data=src)


def test_a_damaged_file_leaves_the_old_model_in_place(tmp_path):
    zip_path, _ = _export(tmp_path)
    user = tmp_path / "user"
    _fake_model(user)
    before = {f: (user / f).read_bytes() for f in model_bundle.model_files()}
    # the same zip with one file swapped after its checksum was taken
    damaged = tmp_path / "damaged.zip"
    victim = model_bundle.model_files()[-1]
    with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(damaged, "w") as dst:
        for info in src.infolist():
            dst.writestr(info, b"tampered" if info.filename == victim else src.read(info))
    with pytest.raises(ModelBundleError, match="damaged"):
        model_bundle.install_bundle(damaged.read_bytes(), data=user)
    assert {f: (user / f).read_bytes() for f in model_bundle.model_files()} == before


def _serve(monkeypatch, zip_path, calls):
    def get(method, url, deadline_s, **kw):
        calls.append(url)
        if url == CFG["model_manifest_url"]:
            return 200, (zip_path.parent / "manifest.json").read_bytes()
        return 200, zip_path.read_bytes()
    monkeypatch.setattr(model_bundle, "request_with_deadline", get)


def test_update_downloads_once_then_waits(tmp_path, monkeypatch):
    zip_path, manifest = _export(tmp_path)
    calls = []
    _serve(monkeypatch, zip_path, calls)
    user = tmp_path / "user"
    got, note = model_bundle.update_if_newer(data=user)
    assert got["created"] == manifest["created"] and note == ""
    assert calls == [CFG["model_manifest_url"], CFG["model_bundle_url"]]
    model_bundle.update_if_newer(data=user)           # checked moments ago
    assert len(calls) == 2
    os.utime(user / model_bundle.CHECK_STAMP, (time.time() - 7 * 3600,) * 2)
    model_bundle.update_if_newer(data=user)           # same snapshot: manifest only
    assert calls[2:] == [CFG["model_manifest_url"]]


def test_a_local_model_is_not_overwritten_until_asked(tmp_path, monkeypatch):
    zip_path, _ = _export(tmp_path)
    calls = []
    _serve(monkeypatch, zip_path, calls)
    user = tmp_path / "user"
    model_bundle.mark_local(user)
    assert model_bundle.update_if_newer(data=user) == (None, "") and calls == []
    model_bundle.update_if_newer(data=user, force=True)
    assert not model_bundle.is_local(user) and len(calls) == 2


def test_offline_is_soft(tmp_path, monkeypatch):
    def down(*a, **kw):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(model_bundle, "request_with_deadline", down)
    got, note = model_bundle.update_if_newer(data=tmp_path / "user")
    assert got is None and "Could not download" in note
    with pytest.raises(ModelBundleError):
        model_bundle.update_if_newer(data=tmp_path / "user", force=True)


def test_a_snapshot_for_a_newer_app_asks_for_an_update(tmp_path, monkeypatch):
    monkeypatch.setattr(model_bundle, "MIN_APP_VERSION", "99.0")
    zip_path, _ = _export(tmp_path)
    got, note = model_bundle.update_if_newer(data=tmp_path / "user", source=zip_path)
    assert "Update Zandmotor" in note
