import os
import sys

import pytest
import requests

from zandmotor import cli, setup_wizard, wind
from zandmotor.errors import AlreadyRunningError
from zandmotor.html_report.render import warnings_html


@pytest.mark.parametrize("argv", [["--step", "0"], ["--step", "7"], ["--step", "x"],
                                  ["--hours", "0"], ["--hours", "500"]])
def test_bad_arguments_are_refused_up_front(argv, capsys):
    with pytest.raises(SystemExit):
        cli._parse_args(argv)
    err = capsys.readouterr().err
    assert "should be" in err or "not a whole number" in err


def test_good_arguments_pass():
    args = cli._parse_args(["--step", "15", "--hours", "12"])
    assert args.step == 15 and args.hours == 12


def test_a_crash_becomes_a_plain_message_and_a_log(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "USER_DIR", tmp_path)

    def boom(args):
        raise ValueError("bad thing")
    monkeypatch.setattr(cli, "_main", boom)
    with pytest.raises(SystemExit) as e:
        cli.main(["--no-open"])
    assert "Something went wrong" in str(e.value) and "zandmotor.log" in str(e.value)
    assert "Traceback" in (tmp_path / "zandmotor.log").read_text()


def test_no_internet_gets_its_own_message(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "USER_DIR", tmp_path)

    def offline(args):
        raise requests.ConnectionError("Name or service not known")
    monkeypatch.setattr(cli, "_main", offline)
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert "internet connection" in str(e.value)


def test_verbose_shows_the_real_error(monkeypatch):
    def boom(args):
        raise ValueError("bad thing")
    monkeypatch.setattr(cli, "_main", boom)
    with pytest.raises(ValueError):
        cli.main(["-v"])


def test_only_one_run_at_a_time(tmp_path):
    lock = tmp_path / ".lock"
    with cli._run_lock(lock):
        with pytest.raises(AlreadyRunningError):
            with cli._run_lock(lock):
                pass
    with cli._run_lock(lock):               # released again
        pass


def test_missing_wind_hours_are_dropped(monkeypatch):
    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"hourly": {"time": ["2026-11-17T10:00", "2026-11-17T11:00",
                                        "2026-11-17T12:00"],
                               "wind_speed_10m": [12, 14, None],
                               "wind_gusts_10m": [18, None, None],
                               "wind_direction_10m": [270, 280, None],
                               "temperature_2m": [9, 9, None], "precipitation": [0, 0, None]},
                    "daily": {}}
    monkeypatch.setattr(requests, "get", lambda *a, **kw: R())
    t, spd, gst, *_ = wind.fetch_wind(52.0, 4.2)
    assert len(t) == 1 and spd[0] == 12 and gst[0] == 18


def test_launchers_run_this_interpreter(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_wizard, "USER_DIR", tmp_path / "Zandmotor Kite")
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    setup_wizard.install_launchers(desktop=desktop)
    text = (tmp_path / "Zandmotor Kite" / "Zandmotor.command").read_text()
    assert sys.executable in text and "-m zandmotor" in text and "__ZANDMOTOR__" not in text
    assert os.access(tmp_path / "Zandmotor Kite" / "Change settings.command", os.X_OK)
    assert (desktop / "Zandmotor.command").resolve() == \
        (tmp_path / "Zandmotor Kite" / "Zandmotor.command").resolve()


def test_warnings_are_escaped_and_folded():
    out = warnings_html(["a <b>", "two", "three", "four"])
    assert "a &lt;b&gt;" in out and "<details" in out and "2 more notes" in out
    assert "<details" not in warnings_html(["one", "two", "three"])
