"""Create slide figures from raw W&B JSON exports; never count repeated logs as seeds.

Usage (PowerShell or Bash):
python scripts/plot_packet_loss_results.py --input results/battle25_packet_loss --out output/packet_loss_figures
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


BLUE = "#2563A6"
ORANGE = "#D97732"
INK = "#172B45"
MUTED = "#586A80"
GRID = "#E4EAF1"


def read_data(directory):
    groups = defaultdict(list)
    identities = set()
    for path in sorted((directory / "runs").glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        config, summary = run["config"], run["summary"]
        identity = {
            key: config.get(key) for key in
            ("checkpoint_path", "load_step", "name", "env", "env_args", "seed",
             "test_nepisode", "test_greedy", "evaluation_epsilon", "agent",
             "topk_neighbors", "one_peer", "packet_loss_train", "packet_loss_test", "evaluate")
        }
        identity["actual_timestep"] = summary.get("timestep")
        identities.add(json.dumps(identity, sort_keys=True))
        groups[config["packet_loss_percent"]].append((path, run))
    if not groups:
        raise ValueError("No raw run JSON files found")
    if len(identities) != 1:
        raise ValueError("Different checkpoint/seed/evaluation configs: inspect runs before plotting")
    rows, audit = [], []
    for loss, runs in sorted(groups.items()):
        signatures = {
            json.dumps({k: v for k, v in run["summary"].items() if k.startswith("test_")}, sort_keys=True)
            for _, run in runs
        }
        if len(signatures) != 1:
            raise ValueError(f"Non-identical results at loss {loss}; do not silently collapse runs")
        representative = runs[0][1]
        summary, config = representative["summary"], representative["config"]
        rows.append(dict(
            loss_percent=loss, win_rate_percent=summary["test_red_team_win_mean"] * 100,
            return_mean=summary["test_return_mean"], return_episode_std=summary["test_return_std"],
            red_team_alives=summary["test_red_team_alives_mean"],
            episode_length=summary["test_episode_length_mean"],
            config_seed=config["seed"], episodes=config["test_nepisode"],
            repeated_logs=len(runs), unique_metric_sets=len(signatures),
            representative_run_id=representative["run_id"],
        ))
        audit.append(dict(loss_percent=loss, run_ids=[run["run_id"] for _, run in runs],
                          source_files=[str(path.resolve()) for path, _ in runs],
                          unique_metric_sets=len(signatures), actual_seed=config["seed"]))
    if [r["loss_percent"] for r in rows] != [0, 10, 20, 30, 50]:
        raise ValueError("These slide annotations require loss rates 0, 10, 20, 30, 50")
    return rows, audit


def canvas(title, subtitle, panels=2):
    fig, axes = plt.subplots(1, panels, figsize=(16, 9), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=.085, right=.96, bottom=.245, top=.74, wspace=.28)
    fig.text(.065, .925, title, fontsize=26, weight="bold", color=INK, va="top")
    fig.text(.065, .86, subtitle, fontsize=15, color=MUTED, va="top")
    fig.text(.065, .073,
             "Descriptive evaluation of one checkpoint. Repeated logs share actual seed 0; no between-seed CI.",
             fontsize=12, color=MUTED)
    return fig, list(axes[0])


def style(ax, ylabel, ylim):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(axis="both", labelsize=13, length=0, pad=9, colors=MUTED)
    ax.set_xticks([0, 10, 20, 30, 50], ["0", "10", "20", "30", "50"])
    ax.set_xlabel("Packet loss on remote messages (%)", fontsize=13, labelpad=14, color=INK)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=12, color=INK)
    ax.set_xlim(-4, 54)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def line(ax, xs, ys, label_format):
    ax.plot(xs, ys, color=BLUE, linewidth=2.6, zorder=3)
    ax.scatter(xs, ys, color=[BLUE] * (len(xs)-1) + [ORANGE], s=90, zorder=4,
               edgecolors="white", linewidths=1.5)
    for x, y in zip(xs, ys):
        ax.annotate(label_format.format(y), (x, y), xytext=(0, 13), textcoords="offset points",
                    ha="center", fontsize=14, weight="bold", color=ORANGE if x == 50 else INK)


def save(fig, directory, stem):
    fig.savefig(directory / (stem + ".png"), dpi=240, facecolor="white")
    fig.savefig(directory / (stem + ".svg"), facecolor="white")
    plt.close(fig)


def create_figures(rows, directory):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    base, last = rows[0], rows[-1]
    xs = [r["loss_percent"] for r in rows]
    win_drop = base["win_rate_percent"] - last["win_rate_percent"]
    reward_drop = 100 * (1 - last["return_mean"] / base["return_mean"])
    length_rise = 100 * (last["episode_length"] / base["episode_length"] - 1)
    alive_drop = 100 * (1 - last["red_team_alives"] / base["red_team_alives"])
    spread_ratio = last["return_episode_std"] / base["return_episode_std"]
    context = (f"ExpoComm one-peer n6 | Battle25 | checkpoint 5,009,240 | "
               f"{base['episodes']} episodes per condition | actual seed {base['config_seed']}")

    fig, axes = canvas("Packet loss reduces win rate and return", context)
    style(axes[0], "Red-team win rate (%)", (0, 112))
    line(axes[0], xs, [r["win_rate_percent"] for r in rows], "{:.0f}%")
    style(axes[1], "Mean episode return", (0, 5.6))
    line(axes[1], xs, [r["return_mean"] for r in rows], "{:.3f}")
    fig.text(.085, .135, f"At 50% loss: win rate −{win_drop:.0f} percentage points", fontsize=16, weight="bold", color=ORANGE)
    fig.text(.575, .135, f"Mean return −{reward_drop:.1f}% vs. no loss", fontsize=16, weight="bold", color=ORANGE)
    save(fig, directory, "01_win_rate_and_return")

    fig, axes = canvas("Fewer survivors and longer episodes at high loss", context)
    style(axes[0], "Mean surviving red agents", (0, 20))
    line(axes[0], xs, [r["red_team_alives"] for r in rows], "{:.2f}")
    style(axes[1], "Mean episode length (steps)", (0, 200))
    line(axes[1], xs, [r["episode_length"] for r in rows], "{:.1f}")
    fig.text(.085, .135, f"At 50% loss: survivors −{alive_drop:.1f}% vs. no loss", fontsize=16, weight="bold", color=ORANGE)
    fig.text(.575, .135, f"Episode length +{length_rise:.1f}% vs. no loss", fontsize=16, weight="bold", color=ORANGE)
    save(fig, directory, "02_survivors_and_episode_length")

    fig, axes = canvas("Packet loss increases return variability", context, panels=1)
    ax = axes[0]
    style(ax, "Standard deviation of episode return", (0, 2.15))
    values = [r["return_episode_std"] for r in rows]
    ax.bar(xs, values, width=5.5, color=[BLUE] * 4 + [ORANGE], zorder=3)
    for x, value in zip(xs, values):
        ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=17, weight="bold", color=INK)
    fig.text(.085, .135, f"At 50% loss: episode-return SD is {spread_ratio:.2f}× the no-loss value.",
             fontsize=17, weight="bold", color=ORANGE)
    fig.text(.085, .108, "Within-run episode spread; not an error bar or uncertainty across evaluation seeds.",
             fontsize=12, color=MUTED)
    save(fig, directory, "03_episode_return_variability")
    return dict(win_drop_pp=win_drop, return_drop_percent=reward_drop,
                survivors_drop_percent=alive_drop, episode_length_increase_percent=length_rise,
                episode_return_std_ratio=spread_ratio)


def write_captions(rows, changes, directory):
    lines = [
        "# Packet-loss evaluation: figure captions and slide notes", "",
        "Các hình có nhãn tiếng Anh, tỷ lệ 16:9. PNG: 3840 × 2160; SVG: vector có thể phóng to.", "",
        "## Dữ liệu và giới hạn", "",
        "Đã đọc toàn bộ 28 file JSON của W&B. Các mức 0%, 10%, 20% có 6 bản ghi/mức; "
        "30%, 50% có 5 bản ghi/mức. Toàn bộ config.seed và env_args.seed đều bằng 0. "
        "Các metric test_* trong từng mức loss giống hệt nhau. Vì vậy, mỗi mức loss chỉ được "
        "biểu diễn bằng một bộ số liệu, không coi các bản ghi lặp là các seed độc lập.", "",
        "Mỗi run khai báo 100 episode, cùng checkpoint 5.009.240, Battle25, ExpoComm one-peer n6. "
        "Không cộng các run lặp thành 500–600 episode độc lập. Không vẽ CI/SD giữa seed. "
        "Đường nối chỉ giúp đọc xu hướng giữa các mức đã đo, không phải kết quả tại mọi mức loss.", "",
        "## 01 — Win rate và reward", "",
        "**Chú thích dưới hình:** Ảnh hưởng của xác suất mất toàn bộ message trên các liên kết từ "
        "agent khác tới hiệu năng ExpoComm. Mỗi điểm biểu diễn kết quả 100 episode của một "
        "checkpoint với seed thực tế bằng 0; các run trùng số liệu được gộp để hiển thị.", "",
        f"**Thông điệp khi thuyết trình:** Win rate giữ ở mức 90–92% khi loss từ 10% đến 30%, "
        f"nhưng giảm còn 67% tại loss 50%. So với không mất message, win rate giảm "
        f"{changes['win_drop_pp']:.0f} điểm phần trăm và reward giảm {changes['return_drop_percent']:.1f}%. "
        "Chưa đủ bằng chứng để kết luận 20% tốt hơn 10% hoặc xác định một ngưỡng suy giảm chính xác.", "",
        "## 02 — Agent sống sót và thời lượng episode", "",
        "**Chú thích dưới hình:** Số agent đỏ sống sót trung bình và số bước trung bình mỗi episode "
        "theo mức packet loss. Thời lượng sử dụng metric test_episode_length_mean của môi trường.", "",
        f"**Thông điệp khi thuyết trình:** Khi loss tăng từ 0% lên 50%, số agent sống sót giảm "
        f"từ 17,22 xuống 10,20 (−{changes['survivors_drop_percent']:.1f}%), trong khi thời lượng "
        f"episode tăng từ 100,72 lên 167,92 bước (+{changes['episode_length_increase_percent']:.1f}%). "
        "Kết hợp với reward/win rate giảm, kết quả phù hợp với sự suy giảm hiệu quả phối hợp. "
        "Đây là diễn giải từ các metric, chưa phải phép đo trực tiếp chất lượng phối hợp hay thời gian thắng; "
        "thời lượng trung bình bao gồm cả trận thắng và trận thua.", "",
        "## 03 — Độ phân tán reward giữa episode", "",
        "**Chú thích dưới hình:** Độ lệch chuẩn reward giữa các episode trong từng run "
        "(test_return_std). Chỉ số này đo mức phân tán kết quả trận đấu, không phải sai số của "
        "reward trung bình hay độ lệch chuẩn giữa các seed.", "",
        f"**Thông điệp khi thuyết trình:** Độ lệch chuẩn reward tăng từ 0,405 lên 1,739, tương đương "
        f"{changes['episode_return_std_ratio']:.2f} lần, khi loss tăng từ 0% lên 50%. "
        "Các trận có kết quả phân tán hơn trong cấu hình đánh giá này.", "",
        "## Bảng số liệu chính xác", "",
        "| Loss | Win rate (%) | Reward mean | Episode reward SD | Red survivors | Episode length |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['loss_percent']:g}% | {row['win_rate_percent']:.0f} | {row['return_mean']:.4f} | "
                     f"{row['return_episode_std']:.4f} | {row['red_team_alives']:.2f} | {row['episode_length']:.2f} |")
    lines += ["", "**Nguồn:** JSON W&B export trong results/battle25_packet_loss/runs. "
              "source_audit.json lưu danh sách run và đường dẫn nguồn; plot_data.csv lưu số liệu được vẽ."]
    (directory / "CAPTIONS_VI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/battle25_packet_loss"))
    parser.add_argument("--out", type=Path, default=Path("output/packet_loss_figures"))
    args = parser.parse_args()
    rows, audit = read_data(args.input)
    args.out.mkdir(parents=True, exist_ok=True)
    changes = create_figures(rows, args.out)
    write_captions(rows, changes, args.out)
    with (args.out / "plot_data.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "source_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(dict(conditions=len(rows), source_runs=sum(row["repeated_logs"] for row in rows),
                          actual_seeds=sorted({row["config_seed"] for row in rows}), changes=changes), indent=2))
    print(f"Saved figures and captions to {args.out.resolve()}")


if __name__ == "__main__":
    main()
