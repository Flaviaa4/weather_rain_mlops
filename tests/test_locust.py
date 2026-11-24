import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def real_locust_installed():
    probe = subprocess.run([sys.executable, "-c", "from locust import HttpUser"], cwd=ROOT, capture_output=True)
    return probe.returncode == 0


needs_locust = pytest.mark.skipif(not real_locust_installed(),
                                  reason="the locust package is not installed, only the repo folder named locust")


def load_summarize():
    spec = importlib.util.spec_from_file_location("summarize", ROOT / "locust" / "summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_stats(folder, tag, requests, failures, rps, p95):
    row = {"Type": "", "Name": "Aggregated", "Request Count": requests, "Failure Count": failures, "Requests/s": rps,
           "50%": p95 / 2, "95%": p95, "99%": p95 * 1.5, "Average Response Time": p95 / 2}
    pd.DataFrame([{**row, "Type": "POST", "Name": "/predict"}, row]).to_csv(folder / f"{tag}_stats.csv", index=False)
    pd.DataFrame({"hostname": ["a", "b"], "responses": [60, 40]}).to_csv(folder / f"{tag}_hostnames.csv", index=False)


def test_summary_table_charts_and_breaking_point(tmp_path):
    write_stats(tmp_path, "r1_u10_rep1", 1000, 0, 9.5, 300)
    write_stats(tmp_path, "r1_u50_rep1", 1000, 50, 20.0, 2500)
    write_stats(tmp_path, "r2_u10_rep1", 1000, 0, 9.8, 150)
    write_stats(tmp_path, "r2_u50_rep1", 1000, 0, 38.0, 900)
    table = load_summarize().write_summary(tmp_path)
    assert len(table) == 4 and set(table["replicas"]) == {1, 2}
    summary = (tmp_path / "summary.md").read_text()
    assert "1 api containers: breaks at 50 users" in summary and "2 api containers: no break" in summary
    assert "2 hosts, 60 / 40" in summary
    assert (tmp_path / "latency_vs_replicas.png").stat().st_size > 0 and (tmp_path / "throughput_vs_replicas.png").exists()


@needs_locust
def test_locustfile_loads_samples_and_synthetic_png(tmp_path):
    code = ("import io, sys; sys.path.insert(0, 'locust'); import locustfile; from PIL import Image; "
            "Image.open(io.BytesIO(locustfile.synthetic_png())).verify(); "
            "print(len(locustfile.load_samples(sys.argv[1])), locustfile.load_samples(sys.argv[1])[0][0])")
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split()[-2:] == ["1", "synthetic.png"]
