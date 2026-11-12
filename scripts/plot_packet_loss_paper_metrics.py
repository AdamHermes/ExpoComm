"""Report packet-loss results using only the MAgent metrics in paper Figure 4.

Battle: Evaluation Win Rate in [0, 1]. AdvPursuit: raw Evaluation Return.
The x-axis is loss probability for our added experiment, not training time.
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_packet_loss_comparison import (
    read_exports, figure, style, curve, save, COLORS, INK, MUTED, ORDER, COUNTS, RATES,
)


def plot(data, out):
    fig, axes = figure("Packet-loss evaluation using the paper's MAgent metrics",
                      "Figure 4 metric convention | ExpoComm one-peer (K=1) | measured results from our checkpoint evaluations")
    for ax, names, metric, ylabel, limits, title in [
        (axes[0, 0], ORDER[3:], "return_mean", "Evaluation Return", (0, 110), "Adversarial Pursuit"),
        (axes[0, 1], ORDER[:3], "win_rate", "Evaluation Win Rate", (0, 1.1), "Battle"),
    ]:
        style(ax, ylabel, limits, title)
        for name, color in zip(names, COLORS):
            curve(ax, data[name], metric, color, f"{name} (N={COUNTS[name]})")
            last = data[name][-1][metric]
            ax.annotate(f"{last:.2f}", (50, last), xytext=(4, 7), textcoords="offset points",
                        fontsize=12, weight="bold", color=color)
        if metric == "win_rate":
            ax.set_yticks(np.arange(0, 1.01, .2))
        else:
            ax.set_yticks([0, 25, 50, 75, 100])
        ax.legend(loc="lower left", frameon=False, fontsize=12)
    fig.text(.08, .125, "Battle35 win rate: 0.89 → 0.48; Battle40: 1.00 → 0.98 as packet loss rises from 0% to 50%.",
             fontsize=15, weight="bold", color=INK)
    save(fig, out, "01_paper_metrics_overview")

    fig, axes = figure("Packet-loss results by map: the same metrics as Figure 4",
                      "Top: Evaluation Return for Adversarial Pursuit. Bottom: Evaluation Win Rate for Battle.", rows=2, cols=3)
    fig.subplots_adjust(bottom=.17, top=.76, hspace=.75)
    for index, name in enumerate(ORDER[3:] + ORDER[:3]):
        ax = axes.flat[index]
        pursuit = name.startswith("AdvPursuit")
        metric = "return_mean" if pursuit else "win_rate"
        style(ax, "Evaluation Return" if pursuit else "Evaluation Win Rate",
              (0, 112) if pursuit else (0, 1.14),
              f"{name} · N={COUNTS[name]}", small=True)
        ax.set_yticks([0, 25, 50, 75, 100] if pursuit else np.arange(0, 1.01, .2))
        curve(ax, data[name], metric, COLORS[index % 3], values=True, fmt="{:.2f}", small=True)
    save(fig, out, "02_paper_metrics_per_map")

    fig, axes = figure("Packet-loss evaluation: reported values",
                      "Raw evaluation metrics from W&B summaries | no normalization to the no-loss baseline", rows=2, cols=1)
    fig.subplots_adjust(top=.75, bottom=.18, hspace=.55)
    for ax, names, metric, title in [
        (axes[0, 0], ORDER[3:], "return_mean", "Adversarial Pursuit — Evaluation Return"),
        (axes[1, 0], ORDER[:3], "win_rate", "Battle — Evaluation Win Rate (0–1)"),
    ]:
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=15, weight="bold", color=INK, pad=16)
        cells = [[f"{name} (N={COUNTS[name]})"] + [f"{r[metric]:.2f}" for r in data[name]] for name in names]
        table = ax.table(cellText=cells, colLabels=["Map"] + [f"Loss {r}%" for r in RATES],
                         colWidths=[.34] + [.132] * 5, cellLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#E2E9F0")
            if row == 0:
                cell.set_facecolor("#EAF0F7")
                cell.set_text_props(color=INK, weight="bold")
            else:
                cell.set_facecolor("white" if row % 2 else "#F7F9FC")
                cell.set_text_props(color=INK)
                if column == 0:
                    cell.set_text_props(ha="left")
    save(fig, out, "03_reported_values_table")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remaining", type=Path)
    parser.add_argument("--battle25", type=Path, default=Path("results/battle25_packet_loss"))
    parser.add_argument("--out", type=Path, default=Path("output/packet_loss_paper_metrics"))
    args = parser.parse_args()
    if args.remaining is None:
        args.remaining = sorted(Path("results").glob("packet_loss_remaining_*"))[-1]
    data, audit = read_exports(args.remaining, args.battle25)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    plot(data, args.out)
    rows = []
    for name in ORDER[3:] + ORDER[:3]:
        pursuit = name.startswith("AdvPursuit")
        metric = "return_mean" if pursuit else "win_rate"
        for record in data[name]:
            rows.append(dict(map=name, agents=COUNTS[name], loss_percent=record["loss_percent"],
                             metric="Evaluation Return" if pursuit else "Evaluation Win Rate",
                             wandb_metric="test_return_mean" if pursuit else "test_red_team_win_mean",
                             value=record[metric], actual_seed=0, episodes=100,
                             checkpoint_step=record["checkpoint_step"]))
    with (args.out / "reported_values.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    notes = ["# Packet loss — báo cáo đúng metric của paper", "",
             "Nguồn định nghĩa metric: Figure 4, trang 8, 2502.19717v1.pdf.", "",
             "- Battle: Evaluation Win Rate, thang 0–1, lấy test_red_team_win_mean.",
             "- AdvPursuit: Evaluation Return gốc, lấy test_return_mean.",
             "- Đây là số đo sweep của nhóm; không phải số lấy từ đường cong trong paper.",
             "- Trục x là packet loss của thí nghiệm bổ sung; Figure 4 gốc dùng training timesteps.",
             "- 100 episode/điều kiện, actual seed 0, một checkpoint/map. Không có CI giữa seed.",
             "- Các bản ghi Battle25 lặp được gộp; không tính thành nhiều seed độc lập.", "",
             "## Chú thích hình 01", "",
             "Ảnh hưởng của packet loss tới Evaluation Return trên AdvPursuit và Evaluation Win Rate "
             "trên Battle, dùng cùng metric như Figure 4 của paper. Trên các checkpoint này, "
             "Battle35 giảm win rate từ 0,89 xuống 0,48, Battle25 từ 1,00 xuống 0,67 và "
             "Battle40 từ 1,00 xuống 0,98 khi loss tăng từ 0% lên 50%. "
             "Evaluation Return của AdvPursuit45/60/70 ở loss 50% lần lượt là 85,14 / 76,03 / 65,70.", "",
             "## Chú thích hình 02", "",
             "Sáu map được trình bày theo bố cục Figure 4: hàng trên là AdvPursuit với Evaluation Return; "
             "hàng dưới là Battle với Evaluation Win Rate. Nhãn N là số learning agents, không phải kích thước map.", "",
             "## Chú thích hình 03", "",
             "Bảng kết quả trực tiếp từ W&B summary, làm tròn hai chữ số thập phân. "
             "reported_values.csv giữ số gốc chưa làm tròn. Không báo cáo normalized return, "
             "survival, episode length hay episode reward SD trong bộ hình này.", "",
             "| Map | Metric | Loss 0% | Loss 10% | Loss 20% | Loss 30% | Loss 50% |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name in ORDER[3:] + ORDER[:3]:
        pursuit = name.startswith("AdvPursuit")
        metric = "return_mean" if pursuit else "win_rate"
        notes.append(f"| {name} | {'Evaluation Return' if pursuit else 'Evaluation Win Rate'} | " +
                     " | ".join(f"{r[metric]:.2f}" for r in data[name]) + " |")
    (args.out / "CAPTIONS_VI.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    (args.out / "source_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    archive = args.out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(args.out.iterdir()):
            bundle.write(path, args.out.name + "/" + path.name)
    print(f"Saved paper-metric figures and 30 original measured values to {args.out.resolve()}")
    print(f"Archive: {archive.resolve()}")


if __name__ == "__main__":
    main()
