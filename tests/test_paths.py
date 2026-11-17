from zandmotor import config
from zandmotor.config import resolve_layout


def test_this_repo_is_a_checkout_and_keeps_the_old_layout():
    assert config.IS_CHECKOUT
    assert config.DATA == config.USER_DIR == config.CODE_ROOT
    assert config.CACHE == config.CODE_ROOT / "cache"


def test_installed_layout_and_home_override(tmp_path):
    pkg_root = tmp_path / "site-packages"
    pkg_root.mkdir()
    is_checkout, user_dir, data = resolve_layout(pkg_root, env={}, home=tmp_path)
    assert not is_checkout
    assert user_dir == tmp_path / "Zandmotor Kite"
    assert "zandmotor" in str(data) and data != user_dir
    _, user_dir, _ = resolve_layout(pkg_root, env={"ZANDMOTOR_HOME": str(tmp_path / "x")})
    assert user_dir == tmp_path / "x"


def test_a_folder_needs_both_git_and_pyproject_to_be_a_checkout(tmp_path):
    (tmp_path / ".git").mkdir()
    assert not resolve_layout(tmp_path, env={})[0]
    (tmp_path / "pyproject.toml").write_text("")
    assert resolve_layout(tmp_path, env={})[0]
