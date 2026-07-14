import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_requirements_matches_pyproject_pin():
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").split()
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert requirements == ["orbitquant>=0.6.0"]
    assert '"orbitquant>=0.6.0"' in pyproject


def _run_install_hook(tmp_path, *, fake_orbitquant: bool, kernels_exit: int = 0):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    if not fake_orbitquant:
        # Shadow any orbitquant installed in the test environment with a stub
        # that fails to import, emulating a runtime without the dependency.
        package_dir = site_dir / "orbitquant"
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text(
            'raise ImportError("orbitquant absent for the install hook test")\n',
            encoding="utf-8",
        )
    if fake_orbitquant:
        package_dir = site_dir / "orbitquant"
        cli_dir = package_dir / "cli"
        cli_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (cli_dir / "__init__.py").write_text("", encoding="utf-8")
        report = {
            "source": "release" if kernels_exit == 0 else "unavailable",
            "variant": "torch-stable-abi211-cpu-x86_64-linux" if kernels_exit == 0 else None,
            "detail": "test detail",
        }
        (cli_dir / "main.py").write_text(
            "import json\n"
            "import sys\n"
            f"print(json.dumps({report!r}))\n"
            f"sys.exit({kernels_exit})\n",
            encoding="utf-8",
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(site_dir)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "install.py")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_install_hook_reports_provisioned_kernels(tmp_path):
    result = _run_install_hook(tmp_path, fake_orbitquant=True, kernels_exit=0)

    assert result.returncode == 0
    assert "source=release" in result.stdout
    assert "variant=torch-stable-abi211-cpu-x86_64-linux" in result.stdout


def test_install_hook_never_fails_when_no_variant_matches(tmp_path):
    result = _run_install_hook(tmp_path, fake_orbitquant=True, kernels_exit=1)

    assert result.returncode == 0
    assert "Triton or dequantized fallbacks" in result.stderr
    assert "test detail" in result.stderr


def test_install_hook_never_fails_without_orbitquant(tmp_path):
    result = _run_install_hook(tmp_path, fake_orbitquant=False)

    assert result.returncode == 0
    assert "not importable" in result.stderr


def test_install_hook_emits_json_report_line(tmp_path):
    result = _run_install_hook(tmp_path, fake_orbitquant=True, kernels_exit=0)
    assert result.returncode == 0
    # The hook consumes the CLI JSON without leaking it raw to the log.
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            json.loads(stripped)
