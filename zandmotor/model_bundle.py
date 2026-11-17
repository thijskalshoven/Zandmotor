"""The shared lagoon model: the maintainer's calibrated files as one snapshot.

A new user has no Copernicus account, so can't run --calibrate, and without a
fitted model the report shows the whole outline as wet. So the maintainer
publishes the files --calibrate and --outline produce as a zip on the rolling
"model" GitHub pre-release (scripts/publish-model.sh), and installed copies
download it (update_if_newer).

The AHN tile goes in too, although anyone could fetch it from PDOK: the basin
is built from AHN's no-return gaps and is part of the wetness fingerprint
(runtime.wetness_fingerprint), so a user who downloaded a newer AHN would get
a different fingerprint and the shared model would be ignored.

The snapshot holds paths relative to DATA, the same in both layouts, so it
unpacks over a checkout as well as over an installed copy's cache folder."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import shutil
import time
import zipfile
from importlib import metadata
from pathlib import Path

import requests

from zandmotor.config import CFG, DATA, UTC, log
from zandmotor.errors import ModelBundleError
from zandmotor.http_utils import request_with_deadline

FORMAT = 1                                  # bump when the files' meaning changes
MIN_APP_VERSION = "0.2.0"                   # oldest app that reads this snapshot right;
                                            # raise it with changes the old code would misread
MANIFEST = "manifest.json"
LOCAL_MANIFEST = "model_manifest.json"      # in DATA: which snapshot is installed
CHECK_STAMP = "model_checked"               # in DATA: mtime = last update check


def model_files():
    """The snapshot's files, relative to DATA. Built from the CFG filenames,
    so a renamed file can't fall out of the snapshot unnoticed."""
    cache = CFG["cache_dir"]
    files = [CFG["lagoon_geojson_file"], CFG["lagoon_basin_file"]] + [
        f"{cache}/{CFG[k]}" for k in ("lagoon_response_file", "wetness_frequency_file",
                                      "wetness_confidence_file", "wetness_meta_file",
                                      "fill_drain_file", "ahn_dtm_file",
                                      "flats_dry_shared_file")]
    if CFG["coastal_dem"]:
        files.append(CFG["coastal_dem"])
    return files


def app_version():
    try:
        return metadata.version("zandmotor")
    except metadata.PackageNotFoundError:   # pragma: no cover - running from source
        return "0"


def _version_tuple(v):
    return tuple(int(x) for x in str(v).split(".") if x.isdigit())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _newest_image(data):
    """ISO date of the newest lagoon image in the record, or None."""
    from zandmotor.lagoon_model.runtime import latest_observation
    from zandmotor.lagoon_model.types import model_from_dict
    try:
        model = model_from_dict(json.loads(
            (data / CFG["cache_dir"] / CFG["lagoon_response_file"]).read_text()))
    except (OSError, ValueError, KeyError):
        return None
    newest = latest_observation(model, dt.datetime.now(UTC))
    return dt.datetime.fromtimestamp(newest["t"], UTC).isoformat() if newest else None


def export_bundle(dest_zip, data=DATA):
    """Writes the snapshot zip, and its manifest.json next to it (the small
    file installed copies check before downloading the zip). Returns the
    manifest."""
    data, dest_zip = Path(data), Path(dest_zip)
    missing = [f for f in model_files() if not (data / f).exists()]
    if missing:
        raise ModelBundleError(f"Can't export the model, missing in {data}: "
                               f"{', '.join(missing)}. Run --outline and --calibrate first.")
    manifest = {"format": FORMAT, "created": dt.datetime.now(UTC).isoformat(timespec="seconds"),
                "newest_image": _newest_image(data), "min_app_version": MIN_APP_VERSION,
                "files": {}}
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in model_files():
            path = data / f
            manifest["files"][f] = _sha256(path.read_bytes())
            z.write(path, f)                # keeps the mtime: dem_age_days reads it
        z.writestr(MANIFEST, json.dumps(manifest, indent=1))
    (dest_zip.parent / MANIFEST).write_text(json.dumps(manifest, indent=1))
    log.info("Exported the lagoon model (%d files, newest image %s) to %s",
             len(manifest["files"]), manifest["newest_image"], dest_zip)
    return manifest


def install_bundle(zip_bytes, data=DATA):
    """Unpacks a snapshot into DATA after checking every file against the
    manifest inside it. Files are staged first and swapped in one by one, so
    a bad download never leaves half a model behind. Returns the manifest."""
    data = Path(data)
    try:
        z = zipfile.ZipFile(io.BytesIO(zip_bytes))
        manifest = json.loads(z.read(MANIFEST))
    except (zipfile.BadZipFile, KeyError, ValueError) as e:
        raise ModelBundleError(f"The model download is damaged ({e}).") from e
    if manifest.get("format") != FORMAT:
        raise ModelBundleError(f"The shared model has format {manifest.get('format')}; this "
                               f"version of the app reads format {FORMAT}. Update the app.")
    staging = data / ".staging"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        for name, digest in manifest["files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ModelBundleError(f"The model download holds an unsafe path: {name}")
            body = z.read(name)
            if _sha256(body) != digest:
                raise ModelBundleError(f"The model download is damaged ({name} doesn't match "
                                       "its checksum).")
            out = staging / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(body)
            stamp = time.mktime(z.getinfo(name).date_time + (0, 0, -1))
            os.utime(out, (stamp, stamp))
        for name in manifest["files"]:
            dest = data / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / name, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    (data / LOCAL_MANIFEST).write_text(json.dumps(manifest, indent=1))
    return manifest


def installed_manifest(data=DATA):
    """The manifest of the installed snapshot, or None."""
    try:
        return json.loads((Path(data) / LOCAL_MANIFEST).read_text())
    except (OSError, ValueError):
        return None


def is_local(data=DATA):
    return (Path(data) / CFG["model_marker_local"]).exists()


def mark_local(data=DATA):
    """This copy fitted its own model (--calibrate / --outline): stop the
    shared one from overwriting it."""
    Path(data).mkdir(parents=True, exist_ok=True)
    (Path(data) / CFG["model_marker_local"]).write_text(
        "This copy calibrated its own lagoon model, so the shared one is not downloaded.\n"
        "Run zandmotor --shared-model to switch back.\n")


def _get(url):
    status, body = request_with_deadline("GET", url, CFG["model_request_deadline_s"])
    if status != 200:
        raise ModelBundleError(f"{url} answered HTTP {status}")
    return body


def update_if_newer(data=DATA, force=False, source=None):
    """Brings the shared model up to date. Returns (manifest or None, note):
    the note is a warning for the report, or "" when there is nothing to say.

    Checks at most every model_check_hours, skips a copy marked local, and
    on any failure keeps the model already there - a missed update only
    matters when there is no model at all. With force (--shared-model) it
    always downloads, clears the local mark, and raises ModelBundleError.
    `source` is a local zip to install instead of the published one."""
    data = Path(data)
    current = installed_manifest(data)
    stamp = data / CHECK_STAMP
    if not force:
        if is_local(data):
            return current, ""
        recent = (stamp.exists() and time.time() - stamp.stat().st_mtime
                  < CFG["model_check_hours"] * 3600)
        if recent and current is not None:
            return current, ""
    data.mkdir(parents=True, exist_ok=True)
    try:
        if source:
            manifest = install_bundle(Path(source).read_bytes(), data)
        else:
            remote = json.loads(_get(CFG["model_manifest_url"]))
            have_all = all((data / f).exists() for f in (current or {}).get("files", {}))
            if force or current is None or remote.get("created") != current.get("created") \
                    or not have_all:
                log.info("Downloading the shared lagoon model (published %s)...",
                         remote.get("created"))
                manifest = install_bundle(_get(CFG["model_bundle_url"]), data)
            else:
                manifest = current
        stamp.touch()
    except (requests.RequestException, ValueError, OSError, ModelBundleError) as e:
        if force:
            raise ModelBundleError(f"Could not install the shared lagoon model: {e}") from e
        log.warning("Could not update the shared lagoon model (%s)", e)
        if current is None:
            return None, ("Could not download the shared lagoon model, so the lagoon is shown "
                          "without it. Check the internet connection and run again.")
        return current, ""
    if force:
        (data / CFG["model_marker_local"]).unlink(missing_ok=True)
    note = ""
    if _version_tuple(manifest.get("min_app_version", 0)) > _version_tuple(app_version()):
        note = ("The shared lagoon model needs a newer version of this app. Double-click "
                "\"Update Zandmotor\" in your Zandmotor Kite folder.")
    return manifest, note
