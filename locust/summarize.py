import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TAG = re.compile(r"r(?P<replicas>\d+)_u(?P<users>\d+)_rep(?P<repeat>\d+)_stats\.csv$")
P95_LIMIT_MS = 2000
FAILURE_LIMIT = 0.01
USER_COLORS = ["#9ec5f0", "#5a9be3", "#2a78d6", "#1b4f8f", "#0f2f57"]
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"
TABLE_HEADERS = {"replicas": "containers", "users": "users", "rps": "requests/s", "speedup": "requests/s vs fewest containers",
                 "median_ms": "median ms", "p95_ms": "p95 ms", "failures": "failures", "failure_rate": "failure rate",
                 "requests": "requests", "hostnames": "responses per container"}


def parse_tag(path):
    match = TAG.search(Path(path).name)
    return {key: int(value) for key, value in match.groupdict().items()} if match else None


def aggregated_row(path):
    frame = pd.read_csv(path)
    row = frame[frame["Name"] == "Aggregated"].iloc[0]
    requests, failures = int(row["Request Count"]), int(row["Failure Count"])
    return {"requests": requests, "failures": failures, "failure_rate": failures / requests if requests else 0.0,
            "rps": float(row["Requests/s"]), "median_ms": float(row["50%"]), "p95_ms": float(row["95%"]),
            "p99_ms": float(row["99%"]), "mean_ms": float(row["Average Response Time"])}


def hostname_spread(path):
    if not Path(path).exists():
        return ""
    frame = pd.read_csv(path)
    return f"{len(frame)} hosts, " + " / ".join(str(n) for n in sorted(frame["responses"], reverse=True))


def api_cpu(path):
    if not Path(path).exists():
        return None
    frame = pd.read_csv(path, header=None, names=["name", "cpu", "memory"])
    api = frame[frame["name"].str.contains("-api-")]
    return round(api["cpu"].str.rstrip("%").astype(float).mean(), 1) if len(api) else None


def collect_runs(folder):
    rows = []
    for path in sorted(Path(folder).glob("*_stats.csv")):
        tag = parse_tag(path)
        if tag:
            prefix = str(path)[: -len("_stats.csv")]
            rows.append({**tag, **aggregated_row(path), "hostnames": hostname_spread(prefix + "_hostnames.csv"),
                         "api_cpu_percent": api_cpu(prefix + "_docker_stats.csv")})
    return pd.DataFrame(rows)


def average_repeats(runs):
    numeric = ["requests", "failures", "failure_rate", "rps", "median_ms", "p95_ms", "p99_ms", "mean_ms"]
    grouped = runs.groupby(["replicas", "users"])
    table = grouped[numeric].mean().round(2)
    table["repeats"] = grouped.size()
    table["hostnames"] = grouped["hostnames"].first()
    return table.reset_index()


def is_broken(row):
    return row["failure_rate"] > FAILURE_LIMIT or row["p95_ms"] > P95_LIMIT_MS


def breaking_points(table):
    points = {}
    for replicas, group in table.groupby("replicas"):
        broken = group[group.apply(is_broken, axis=1)].sort_values("users")
        points[int(replicas)] = int(broken["users"].iloc[0]) if len(broken) else None
    return points


def add_speedup(table):
    fewest = table.loc[table.groupby("users")["replicas"].idxmin()].set_index("users")["rps"]
    return table.assign(speedup=(table["rps"] / table["users"].map(fewest)).round(2))


def cell(column, value):
    if column == "failure_rate":
        return f"{value:.2%}"
    return f"{value:g}" if column in {"failures", "requests"} else str(value)


def display_table(table):
    return table[list(TABLE_HEADERS)].rename(columns=TABLE_HEADERS)


def markdown_table(table):
    lines = ["| " + " | ".join(TABLE_HEADERS.values()) + " |", "|" + "---|" * len(TABLE_HEADERS)]
    lines += ["| " + " | ".join(cell(c, row[c]) for c in TABLE_HEADERS) + " |" for _, row in table.iterrows()]
    return "\n".join(lines)


def breaking_text(points):
    return "\n".join(f"- {replicas} api containers: " + (f"breaks at {users} users" if users else "no break in the tested range")
                     for replicas, users in points.items())


def style_axis(axis, ylabel):
    axis.set_xlabel("api containers", color=MUTED)
    axis.set_ylabel(ylabel, color=MUTED)
    axis.grid(axis="y", color=GRID, linewidth=1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=MUTED)


def plot_metric(table, metric, ylabel, title, path):
    figure, axis = plt.subplots(figsize=(6.4, 4))
    colors = USER_COLORS[-table["users"].nunique():]
    for color, (users, group) in zip(colors, table.groupby("users")):
        group = group.sort_values("replicas")
        axis.plot(group["replicas"], group[metric], color=color, linewidth=2, marker="o", markersize=7, label=f"{users} users")
    axis.set_xticks(sorted(table["replicas"].unique()))
    style_axis(axis, ylabel)
    axis.legend(frameon=False, title="load")
    axis.set_title(title, loc="left", color=INK, fontweight="bold")
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return Path(path)


def write_summary(folder):
    folder = Path(folder)
    table = add_speedup(average_repeats(collect_runs(folder)))
    table.to_csv(folder / "summary.csv", index=False)
    plot_metric(table, "p95_ms", "p95 latency (ms)", "p95 latency vs api containers", folder / "latency_vs_replicas.png")
    plot_metric(table, "rps", "requests per second", "Throughput vs api containers", folder / "throughput_vs_replicas.png")
    text = (f"# Flood test summary\n\n{markdown_table(table)}\n\nBreaking point (failures above {FAILURE_LIMIT:.0%} "
            f"or p95 above {P95_LIMIT_MS} ms):\n\n{breaking_text(breaking_points(table))}\n")
    (folder / "summary.md").write_text(text, encoding="utf-8")
    return table


if __name__ == "__main__":
    print(display_table(write_summary(sys.argv[1] if len(sys.argv) > 1 else "locust/results")).to_string(index=False))
