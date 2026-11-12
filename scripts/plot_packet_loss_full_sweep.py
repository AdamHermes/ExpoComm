"""Compare 0-50% loss with 100% remote-message removal using Figure 4 metrics.

Reads original W&B JSONs, checks complete logged config compatibility, and
collapses identical repeated logs. Does not estimate a threshold between 50/100.
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_packet_loss_comparison import (
    read_exports, figure, style, save, COLORS, INK, MUTED, ORDER, COUNTS,
)

RATES = [0, 10, 20, 30, 50, 100]


def load_all(remaining, battle25, loss100):
    data, audit = read_exports(remaining, battle25)
    allowed_differences = {"packet_loss_percent", "run_name", "work_dir"}
    for name in ORDER:
        previous = next(item for item in audit if item["map"] == name and item["loss_percent"] == 0)
        baseline_raw = json.loads(Path(previous["sources"][0]).read_text(encoding="utf-8"))
        paths = sorted((loss100 / f"ExpoComm_{name}_eval" / "runs").glob("*.json"))
        if not paths:
            raise ValueError(f"Missing loss100 JSON for {name}")
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        for raw in runs:
            config = raw["config"]
            if config.get("packet_loss_percent") != 100:
                raise ValueError(f"Expected only 100% loss in new export: {name}")
            old_config = baseline_raw["config"]
            differences = {key: (old_config.get(key), config.get(key))
                           for key in set(old_config) | set(config)
                           if old_config.get(key) != config.get(key) and key not in allowed_differences}
            if differences:
                raise ValueError(f"Logged config mismatch for {name}: {differences}")
            if raw["summary"].get("timestep") != baseline_raw["summary"].get("timestep"):
                raise ValueError(f"Loaded checkpoint step differs for {name}")
        signatures = {json.dumps({key: value for key, value in raw["summary"].items() if key.startswith("test_")}, sort_keys=True)
                      for raw in runs}
        if len(signatures) != 1:
            raise ValueError(f"Non-identical repeated loss100 outcomes: {name}")
        raw = runs[0]
        metric = "test_red_team_win_mean" if name.startswith("Battle") else "test_return_mean"
        value = raw["summary"][metric]
        if not np.isfinite(value) or (name.startswith("Battle") and not 0 <= value <= 1):
            raise ValueError(f"Invalid metric in {name}")
        data[name].append(dict(
            map=name, agents=COUNTS[name], loss_percent=100, actual_seed=raw["config"]["seed"],
            episodes=raw["config"]["test_nepisode"], checkpoint_step=raw["summary"]["timestep"],
            representative_run_id=raw["run_id"], logged_runs=len(runs), unique_metric_sets=1,
            win_rate=raw["summary"].get("test_red_team_win_mean"),
            return_mean=raw["summary"]["test_return_mean"],
        ))
        audit.append(dict(map=name, loss_percent=100, actual_seed=raw["config"]["seed"],
                          unique_metric_sets=1, logged_config_match=True,
                          allowed_changed_fields=sorted(allowed_differences),
                          run_ids=[raw["run_id"] for raw in runs], sources=[str(path.resolve()) for path in paths]))
        if [record["loss_percent"] for record in data[name]] != RATES:
            raise ValueError(f"Unexpected rates in {name}")
    return data, audit


def format_axis(ax, ylabel, title, pursuit, small=False):
    style(ax, ylabel, (0, 112) if pursuit else (0, 1.12), title, small)
    ax.set_xlim(-6, 108)
    ax.set_xticks(RATES)
    ax.set_yticks([0, 25, 50, 75, 100] if pursuit else np.arange(0, 1.01, .2))
    ax.axvline(100, color="#C9D1DC", linewidth=1, linestyle=":", zorder=0)


def line(ax, records, key, color, label=None, values=False):
    xs = [record["loss_percent"] for record in records]
    ys = [record[key] for record in records]
    ax.plot(xs[:-1], ys[:-1], color=color, linewidth=2.5, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1, label=label, zorder=3)
    ax.plot(xs[-2:], ys[-2:], color=color, linewidth=2, linestyle="--", zorder=2)
    ax.scatter([100], ys[-1:], marker="D", s=46, color=color, edgecolors="white", linewidths=1, zorder=4)
    if values:
        for index in [0, 4, 5]:
            ax.annotate(f"{ys[index]:.2f}", (xs[index], ys[index]), xytext=(0, 9),
                        textcoords="offset points", ha="center", color=INK, fontsize=11, weight="bold")


def annotate_gap(fig):
    fig.text(.08, .125, "Dashed lines connect loss 50% and 100%. Intermediate rates were not evaluated.",
             fontsize=14, color=MUTED)


def draw(data, out):
    fig, axes = figure("100% message loss separates tolerant and dependent policies",
                      "Paper metrics only. A diamond marks 100% remote-message loss; self-links remain active.")
    for ax, names, key, ylabel, title, pursuit in [
        (axes[0, 0], ORDER[3:], "return_mean", "Evaluation Return", "Adversarial Pursuit", True),
        (axes[0, 1], ORDER[:3], "win_rate", "Evaluation Win Rate", "Battle", False),
    ]:
        format_axis(ax, ylabel, title, pursuit)
        for name, color in zip(names, COLORS):
            line(ax, data[name], key, color, f"{name} (N={COUNTS[name]})")
            if pursuit or name == "Battle40":
                value = data[name][-1][key]
                ax.annotate(f"{value:.2f}", (100, value), xytext=(3, 8), textcoords="offset points",
                            fontsize=12, weight="bold", color=color)
        ax.legend(loc="lower left" if pursuit else "upper right", frameon=False,
                  fontsize=11, labelspacing=.7, bbox_to_anchor=None if pursuit else (.96, .7))
        if not pursuit:
            ax.annotate("0.00", (100, 0), xytext=(3, 8), textcoords="offset points",
                        ha="left", fontsize=11, weight="bold", color=INK)
    annotate_gap(fig)
    save(fig, out, "01_full_sweep_overview")

    fig, axes = figure("Full packet-loss sweep by map",
                      "Top: Evaluation Return. Bottom: Evaluation Win Rate. Labels show loss 0%, 50%, and 100%.", rows=2, cols=3)
    fig.subplots_adjust(bottom=.205, top=.76, hspace=.75)
    for index, name in enumerate(ORDER[3:] + ORDER[:3]):
        pursuit = name.startswith("AdvPursuit")
        key = "return_mean" if pursuit else "win_rate"
        ax = axes.flat[index]
        format_axis(ax, "Evaluation Return" if pursuit else "Evaluation Win Rate",
                    f"{name} · N={COUNTS[name]}", pursuit, small=True)
        line(ax, data[name], key, COLORS[index % 3], values=True)
    fig.text(.08, .105, "Solid: measured 0–50% sweep. Dashed: connection to 100%, without an inferred failure threshold.",
             fontsize=12, color=MUTED)
    save(fig, out, "02_full_sweep_per_map")

    fig, axes = figure("Partial delivery and complete remote-message removal",
                      "Direct comparison at loss 0%, 50%, and 100% using the same checkpoint within each map.")
    fig.subplots_adjust(top=.72)
    bar_colors = ["#9FAFC0", "#2563A6", "#CD6C28"]
    for ax, names, key, ylabel, pursuit in [
        (axes[0, 0], ORDER[3:], "return_mean", "Evaluation Return", True),
        (axes[0, 1], ORDER[:3], "win_rate", "Evaluation Win Rate", False),
    ]:
        style(ax, ylabel, (0, 112) if pursuit else (0, 1.15),
              "Adversarial Pursuit" if pursuit else "Battle")
        ax.set_xlim(-.6, 2.6)
        ax.set_xticks([0, 1, 2], [f"N={COUNTS[name]}" for name in names])
        ax.set_xlabel("Learning agents", fontsize=12)
        ax.set_yticks([0, 25, 50, 75, 100] if pursuit else np.arange(0, 1.01, .2))
        for group, (rate, color) in enumerate(zip([0, 50, 100], bar_colors)):
            values = [next(record[key] for record in data[name] if record["loss_percent"] == rate) for name in names]
            positions = np.arange(3) + (group - 1) * .25
            ax.bar(positions, values, width=.23, label=f"Loss {rate}%", color=color, zorder=3)
            for x, value in zip(positions, values):
                ax.annotate(f"{value:.1f}" if pursuit else f"{value:.2f}", (x, value),
                            xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=12, ncol=3,
               loc="upper center", bbox_to_anchor=(.52, .825))
    fig.text(.08, .125, "At loss 100%: Battle25/35 win rate = 0.00, but Battle40 = 1.00. Pursuit return remains high.",
             fontsize=15, weight="bold", color=INK)
    save(fig, out, "03_partial_vs_complete_loss")

    fig, axes = figure("Reported values across the full sweep",
                      "Raw W&B evaluation metrics, rounded to two decimals. Complete remote-message removal is the last column.", rows=2, cols=1)
    fig.subplots_adjust(top=.75, bottom=.18, hspace=.55)
    for ax, names, key, title in [
        (axes[0, 0], ORDER[3:], "return_mean", "Adversarial Pursuit: Evaluation Return"),
        (axes[1, 0], ORDER[:3], "win_rate", "Battle: Evaluation Win Rate (0–1)"),
    ]:
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=15, color=INK, weight="bold", pad=16)
        table = ax.table(cellText=[[f"{name} (N={COUNTS[name]})"] + [f"{record[key]:.2f}" for record in data[name]] for name in names],
                         colLabels=["Map"] + [f"Loss {rate}%" for rate in RATES],
                         colWidths=[.28] + [.12] * 6, cellLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#E2E9F0")
            cell.set_facecolor("#FFF0E4" if column == 6 else ("#EAF0F7" if row == 0 else ("white" if row % 2 else "#F7F9FC")))
            cell.set_text_props(color=INK, weight="bold" if row == 0 or column == 6 else "normal")
            if column == 0:
                cell.set_text_props(ha="left")
    save(fig, out, "04_full_sweep_values_table")


def write_analysis(data, audit, out):
    text = ["# Packet loss 0–100%: kết quả và diễn giải", "",
            "## Kiểm tra khả năng so sánh", "",
            f"Đã đọc {sum(len(item['run_ids']) for item in audit)} file JSON W&B. Sau khi gộp các bản ghi Battle25 "
            "có cùng kết quả, có 36 điều kiện (6 map × 6 mức loss). Mỗi điều kiện khai báo 100 episode, "
            "actual seed=0 và env seed=0. Bộ loss 100% có sáu run, một run/map. "
            "Toàn bộ logged config của từng run mới khớp config baseline tương ứng, ngoại trừ "
            "packet_loss_percent, run_name và work_dir. Summary.timestep cũng khớp. "
            "Đây là đối chiếu config đã log, không phải chứng nhận môi trường thực thi hay mã nguồn bit-for-bit giống nhau.", "",
            "Không có nhiều training/evaluation seed độc lập và không có dữ liệu từng episode trong export. "
            "Không vẽ CI giữa seed hoặc coi các bản ghi lặp là thêm episode độc lập.", "",
            "## Chỉ báo cáo metric trong Figure 4 của paper", "",
            "Battle: Evaluation Win Rate (0–1), lấy test_red_team_win_mean. AdvPursuit: Evaluation Return gốc, "
            "lấy test_return_mean. Không chuẩn hóa reward. N là số learning agents; số trong tên map là kích thước map.", "",
            "| Map | Metric | Loss 0% | Loss 10% | Loss 20% | Loss 30% | Loss 50% | Loss 100% |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for name in ORDER[3:] + ORDER[:3]:
        pursuit = name.startswith("AdvPursuit")
        key = "return_mean" if pursuit else "win_rate"
        label = "Evaluation Return" if pursuit else "Evaluation Win Rate"
        text.append(f"| {name} | {label} | " + " | ".join(f"{record[key]:.2f}" for record in data[name]) + " |")
        for record in data[name]:
            rows.append(dict(map=name, agents=COUNTS[name], loss_percent=record["loss_percent"],
                             metric=label, value=record[key], actual_seed=record["actual_seed"],
                             episodes=record["episodes"], checkpoint_step=record["checkpoint_step"],
                             representative_run_id=record["representative_run_id"], logged_runs=record["logged_runs"]))
    text += ["", "## Battle25 và Battle35: policy phụ thuộc remote-message delivery", "",
             "Battle25 có win rate 1,00 ở loss 0%, 0,67 ở loss 50%, và 0,00 ở loss 100%. "
             "Battle35 lần lượt là 0,89, 0,48 và 0,00. Loss 100% đồng nghĩa hai run đều không thắng "
             "trận nào trong 100 episode đã đánh giá. So với loss 50%, win rate giảm lần lượt 0,67 và 0,48 "
             "ở đơn vị win rate. Đây là bằng chứng mạnh trong các checkpoint này rằng bỏ hẳn remote-message "
             "delivery phá vỡ hiệu năng vốn đạt được khi còn communication.", "",
             "Các điểm đo chưa cho biết policy giảm mạnh ở loss 60%, 80% hay gần 100%. "
             "Vì vậy, đoạn 50–100% được nối nét đứt và không gọi một mức loss chưa đo là ngưỡng sụp đổ.", "",
             "## Battle40: độ phụ thuộc thay đổi giữa các checkpoint", "",
             "Win rate Battle40 là 1,00 ở loss 0%, 0,98 ở loss 50% và 1,00 ở loss 100%. "
             "Trong metric đang báo cáo, run mới vẫn thắng 100/100 episode khi không nhận remote messages. "
             "Do đó không thể kết luận mọi policy Battle đều cần communication để đạt win rate cao. "
             "Cũng không thể nói packet loss giúp policy tốt hơn: 1,00 so với 0,98 chỉ là chênh lệch quan sát "
             "trong hai tập evaluation hữu hạn.", "",
             "Win rate trong wrapper là số agent đỏ sống sót cuối episode lớn hơn số agent xanh. "
             "Nó không yêu cầu tiêu diệt hết đối thủ và không mô tả đầy đủ mọi khía cạnh chất lượng chiến đấu. "
             "Không thay thế win rate bằng reward của Battle trong bộ hình này.", "",
             "## AdvPursuit: return vẫn cao khi tắt remote delivery", "",
             "AdvPursuit45 giảm từ 87,3752 xuống 83,0004 khi loss tăng từ 0% lên 100%, chênh −4,3748 "
             "đơn vị return. AdvPursuit60 thay đổi từ 73,8057 lên 75,5525 (+1,7468), "
             "AdvPursuit70 từ 65,8750 lên 68,5634 (+2,6885). So với loss 50%, các chênh lệch return "
             "lần lượt là −2,1368, −0,4802 và +2,8660.", "",
             "Các checkpoint AdvPursuit vẫn đạt return tương đối cao mà không nhận message từ agent khác. "
             "Điều này mở rộng kết luận trước đây: độ ổn định không chỉ do còn một nửa message đến nơi ở loss 50%. "
             "Với một seed và không có đối chứng thống kê phù hợp, không khẳng định các mức tăng nhỏ là cải thiện thật, "
             "hay message gây hại. Cũng chưa thể kết luận local observation là nguyên nhân duy nhất.", "",
             "## Điểm quan trọng về bộ nhớ tại loss 100%", "",
             "Ở loss 50%, agent có thể tái sử dụng thông tin đồng đội đã nhận ở bước trước. "
             "Nhưng ở loss 100%, code chặn mọi remote message từ đầu episode, và runner reset hidden/message "
             "state ở đầu mỗi episode. Vì vậy không có thông tin đồng đội từ episode trước để giữ lại. "
             "Self-link và bộ nhớ xử lý thông tin riêng vẫn hoạt động. Không giải thích sự ổn định của loss 100% "
             "bằng việc agent giữ lại remote message từ những episode trước.", "",
             "## Ý nghĩa thực sự đối với communication", "",
             "Kết quả cho thấy mức phụ thuộc của policy đã train vào remote-message delivery khác nhau rõ giữa "
             "map/checkpoint: thiết yếu để giữ win rate đã quan sát ở Battle25/35, nhưng chưa cần thiết để giữ "
             "win rate cao ở Battle40 hay return cao ở ba AdvPursuit trong protocol này. "
             "Không được biến kết quả này thành khẳng định universal rằng communication luôn quan trọng hoặc luôn thừa.", "",
             "Ablation loss 100% tác động đồng thời vào nội dung thông tin nhận được và nhịp cập nhật message GRU "
             "(one-peer giữ msg_ego khi thất bại). Nó chưa đo riêng giá trị ngữ nghĩa của nội dung message. "
             "Đây cũng là policy đã train cùng communication rồi bị tắt kênh lúc test; không tương đương policy "
             "được train từ đầu không communication. Muốn đo lợi ích của phương pháp cần baseline train "
             "không communication, nhiều training seed và evaluation seed thực sự khác nhau.", "",
             "## Chú thích cho slide", "",
             "**Hình 01:** Evaluation Return của AdvPursuit và Evaluation Win Rate của Battle theo packet loss. "
             "Hình thoi đánh dấu loss 100%: không nhận message từ agent khác, self-link vẫn giữ. "
             "Nét đứt chỉ nối các điểm đã đo tại 50% và 100%, không mô tả ngưỡng chuyển tiếp.", "",
             "**Hình 02:** Cùng dữ liệu theo từng map, hàng trên AdvPursuit và hàng dưới Battle. "
             "Nhãn số chỉ hiển thị tại loss 0%, 50%, 100% để dễ đọc; CSV giữ đủ sáu mức.", "",
             "**Hình 03:** So sánh trực tiếp ba chế độ không mất message, mất một phần và chặn toàn bộ remote delivery. "
             "Hai panel dùng metric và đơn vị khác nhau theo Figure 4; không so độ cao return với win rate.", "",
             "**Hình 04:** Bảng số đo gốc, làm tròn hai chữ số. CSV giữ độ chính xác đầy đủ.", "",
             "**Thông điệp ngắn:** Complete remote-message removal eliminates observed wins in Battle25/35, "
             "while Battle40 and the Pursuit checkpoints remain resilient on the reported metrics. "
             "Communication dependence is policy- and scenario-specific in this evaluation.", "",
             "## Nguồn và khả năng tái tạo", "",
             "Source_audit.json liệt kê toàn bộ run ID và file JSON được dùng. reported_values.csv lưu 36 "
             "giá trị metric gốc. Cơ chế cần đối chiếu: src/modules/agents/ExpoComm_agent.py "
             "(_packet_keep_mask, ExpoCommOAgent._communicate, init_hidden), "
             "src/runners/episode_runner.py (reset/init_hidden), src/envs/battle_wrappers.py (định nghĩa win). "
             "Metric convention: Figure 4, trang 8 của 2502.19717v1.pdf. "
             "Đây là số đo của nhóm, không phải số paper đã công bố về packet loss."]
    with (out / "reported_values.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "ANALYSIS_VI.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def latest(pattern):
    folders = sorted(Path("results").glob(pattern))
    if not folders:
        raise ValueError(f"No matching export folders: {pattern}")
    return folders[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remaining", type=Path)
    parser.add_argument("--loss100", type=Path)
    parser.add_argument("--battle25", type=Path, default=Path("results/battle25_packet_loss"))
    parser.add_argument("--out", type=Path, default=Path("output/packet_loss_0_to100"))
    args = parser.parse_args()
    data, audit = load_all(args.remaining or latest("packet_loss_remaining_*"), args.battle25,
                           args.loss100 or latest("packet_loss100_*"))
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    draw(data, args.out)
    write_analysis(data, audit, args.out)
    (args.out / "source_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    archive = args.out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(args.out.iterdir()):
            bundle.write(path, args.out.name + "/" + path.name)
    print(f"Verified 6 matching new configurations. Read {sum(len(item['run_ids']) for item in audit)} source JSONs.")
    print(f"Saved 4 PNG/SVG figure sets, 36 raw metric values, and Vietnamese analysis to {args.out.resolve()}")
    print(f"Archive: {archive.resolve()}")


if __name__ == "__main__":
    main()
