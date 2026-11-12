"""Make cross-map slide figures directly from downloaded W&B run JSON files.

Defaults: latest results/packet_loss_remaining_* plus the older Battle25 export.
Identical repeated logs are collapsed, never counted as independent seed trials.
"""
import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator


RATES = [0, 10, 20, 30, 50]
ORDER = ["Battle25", "Battle35", "Battle40", "AdvPursuit45", "AdvPursuit60", "AdvPursuit70"]
COUNTS = dict(zip(ORDER, [20, 42, 64, 25, 45, 61]))
COLORS = ["#2563A6", "#CD6C28", "#138477"]
INK, MUTED, GRID = "#182E49", "#52657B", "#E2E9F0"
METRICS = {"return_mean": "test_return_mean", "return_episode_std": "test_return_std",
           "episode_length": "test_episode_length_mean", "red_alive": "test_red_team_alives_mean",
           "win_rate": "test_red_team_win_mean"}


def read_exports(remaining, battle25):
    exports = {name: remaining / f"ExpoComm_{name}_eval" for name in ORDER if name != "Battle25"}
    exports["Battle25"] = battle25
    data, audit = {}, []
    for name in ORDER:
        folder = exports[name]
        groups, identities = defaultdict(list), set()
        for path in sorted((folder / "runs").glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            config, summary = raw["config"], raw["summary"]
            if not config.get("evaluate") or not config.get("packet_loss_test") or config.get("packet_loss_train"):
                raise ValueError(f"Expected test-only evaluation: {path}")
            if config["seed"] != 0 or config["env_args"]["seed"] != 0 or config["test_nepisode"] != 100:
                raise ValueError(f"Figure labels expect actual seed 0 and 100 episodes: {path}")
            identity = {k: config.get(k) for k in
                        ("checkpoint_path", "load_step", "env", "env_args", "name", "agent", "mac",
                         "one_peer", "topk_neighbors", "seed", "test_nepisode", "test_greedy", "evaluation_epsilon")}
            identity["actual_checkpoint_step"] = summary.get("timestep")
            identities.add(json.dumps(identity, sort_keys=True))
            groups[config["packet_loss_percent"]].append((path, raw))
        if len(identities) != 1 or sorted(groups) != RATES:
            raise ValueError(f"Missing rates or mixed evaluation configuration: {folder}")
        records = []
        for loss, runs in sorted(groups.items()):
            signatures = {json.dumps({k: v for k, v in raw["summary"].items() if k.startswith("test_")}, sort_keys=True)
                          for _, raw in runs}
            if len(signatures) != 1:
                raise ValueError(f"Repeated logs have different outcomes for {name} / {loss}%")
            raw = runs[0][1]
            record = dict(map=name, agents=COUNTS[name], loss_percent=loss,
                          actual_seed=0, episodes=100, logged_runs=len(runs),
                          unique_metric_sets=1, representative_run_id=raw["run_id"],
                          checkpoint_step=raw["summary"]["timestep"])
            for key, metric in METRICS.items():
                record[key] = raw["summary"].get(metric)
                value = record[key]
                if value is not None and not math.isfinite(value):
                    raise ValueError(f"Non-finite metric in {name}: {metric}")
            if any(record[key] is None for key in ["return_mean", "return_episode_std", "episode_length"]):
                raise ValueError(f"Missing reward/length metric: {name}")
            if name.startswith("Battle") and (record["win_rate"] is None or record["red_alive"] is None):
                raise ValueError(f"Missing Battle metrics: {name}")
            records.append(record)
            audit.append(dict(map=name, loss_percent=loss, actual_seed=0, unique_metric_sets=1,
                              run_ids=[raw["run_id"] for _, raw in runs],
                              sources=[str(path.resolve()) for path, _ in runs]))
        baseline = records[0]["return_mean"]
        if baseline <= 0:
            raise ValueError(f"Relative-return chart requires a positive baseline: {name}")
        for record in records:
            record["return_retained_percent"] = record["return_mean"] / baseline * 100
            record["return_change_percent"] = record["return_retained_percent"] - 100
            if record["red_alive"] is not None:
                record["red_survival_percent"] = record["red_alive"] / record["agents"] * 100
        data[name] = records
    return data, audit


def figure(title, subtitle, rows=1, cols=2):
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9), squeeze=False)
    fig.subplots_adjust(left=.08, right=.97, bottom=.21, top=.76,
                        wspace=.27, hspace=.60 if rows > 1 else .2)
    fig.text(.06, .94, title, fontsize=25, weight="bold", color=INK, va="top")
    fig.text(.06, .875, subtitle, fontsize=13, color=MUTED, va="top")
    fig.text(.06, .065, "100 episodes/condition | actual seed 0 | one trained checkpoint/map | no between-seed CI",
             fontsize=12, color=MUTED)
    return fig, axes


def style(ax, ylabel, ylim, title=None, small=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(length=0, pad=8, labelsize=11 if small else 13, colors=MUTED)
    ax.set_xticks(RATES)
    ax.set_xlim(-3, 54)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("Packet loss on remote messages (%)", fontsize=10 if small else 12, labelpad=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=11 if small else 13, labelpad=10, color=INK)
    if title:
        ax.set_title(title, fontsize=15, weight="bold", loc="left", pad=15, color=INK)


def curve(ax, records, key, color, label=None, scale=1, values=False, fmt="{:.2f}", small=False):
    xs, ys = [r["loss_percent"] for r in records], [r[key] * scale for r in records]
    ax.plot(xs, ys, color=color, linewidth=2.6, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=1.1, label=label, zorder=3)
    if values:
        for x, y in zip(xs, ys):
            ax.annotate(fmt.format(y), (x, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=10 if small else 13, color=INK, weight="bold")


def save(fig, directory, stem):
    # Check bounds at the figure's actual rendering DPI before writing artifacts.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            bounds = text.get_window_extent(renderer)
            if bounds.x0 < -2 or bounds.y0 < -2 or bounds.x1 > fig.bbox.width + 2 or bounds.y1 > fig.bbox.height + 2:
                raise ValueError(f"Text outside figure bounds in {stem}: {text.get_text()}")
    fig.savefig(directory / (stem + ".png"), dpi=240, facecolor="white")
    fig.savefig(directory / (stem + ".svg"), facecolor="white")
    plt.close(fig)


def draw(data, out):
    battles, pursuits = ORDER[:3], ORDER[3:]
    fig, axes = figure("Packet-loss sensitivity differs across maps and tasks",
                      "Return retained relative to each map's own no-loss baseline (=100%); both panels use the same scale.")
    for ax, names, title in zip(axes[0], [battles, pursuits], ["Battle", "Adversarial Pursuit"]):
        style(ax, "Return retained vs. no loss (%)", (0, 116), title)
        ax.axhline(100, color=MUTED, linestyle="--", linewidth=1)
        for name, color in zip(names, COLORS):
            final = data[name][-1]["return_retained_percent"]
            curve(ax, data[name], "return_retained_percent", color,
                  f"{name}: {final:.1f}% retained at loss 50%")
        ax.legend(loc="lower left", fontsize=11, frameon=False, labelspacing=.8)
    fig.text(.08, .125, "At loss 50%: Battle35 retains 16.8% of baseline return; Pursuit maps retain 97.4–103.0%.",
             fontsize=16, weight="bold", color=INK)
    save(fig, out, "01_relative_return_across_tasks")

    fig, axes = figure("Battle35 has the largest observed win-rate drop",
                      "Win rate follows the environment rule: red survivors > blue survivors at episode end.", cols=3)
    for ax, name, color in zip(axes[0], battles, COLORS):
        style(ax, "Red-team win rate (%)", (0, 117), f"{name} · N={COUNTS[name]}", small=True)
        curve(ax, data[name], "win_rate", color, scale=100, values=True, fmt="{:.0f}%", small=True)
    fig.text(.08, .125, "Loss 0% → 50%: Battle25 100% → 67%   |   Battle35 89% → 48%   |   Battle40 100% → 98%",
             fontsize=15, weight="bold", color=INK)
    save(fig, out, "02_battle_win_rate")

    fig, axes = figure("Mean return falls in Battle; Pursuit changes are smaller",
                      "Absolute returns: shared scale within each task, different scales between rows. Compare sensitivity within each map.", rows=2, cols=3)
    fig.subplots_adjust(bottom=.17, top=.76, hspace=.75)
    for index, name in enumerate(ORDER):
        ax = axes.flat[index]
        style(ax, "Mean episode return", (0, 5.8) if index < 3 else (0, 108),
              f"{name} · N={COUNTS[name]}", small=True)
        curve(ax, data[name], "return_mean", COLORS[index % 3], values=True,
              fmt="{:.2f}" if index < 3 else "{:.1f}", small=True)
    save(fig, out, "03_absolute_return_all_maps")

    fig, axes = figure("Battle survival and episode length depend on the map",
                      "Survivors are normalized by the initial red-team size; duration includes both wins and losses.")
    style(axes[0, 0], "Red agents surviving at end (%)", (0, 107), "Survival fraction")
    style(axes[0, 1], "Mean episode length (environment steps)", (0, 220), "Episode duration")
    for name, color in zip(battles, COLORS):
        curve(axes[0, 0], data[name], "red_survival_percent", color, name)
        curve(axes[0, 1], data[name], "episode_length", color, name)
    for ax in axes[0]:
        ax.legend(loc="lower left", frameon=False, fontsize=12)
    fig.text(.08, .125, "Longer episodes are not universal: Battle35 duration falls slightly while its win rate also falls.",
             fontsize=15, color=INK, weight="bold")
    save(fig, out, "04_battle_survival_and_duration")

    fig, axes = figure("Observed return change at 50% packet loss",
                      "Relative change from the same checkpoint's no-loss baseline; positive values are not evidence of a beneficial effect.", cols=1)
    ax = axes[0, 0]
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    positions = [0, 1, 2, 4, 5, 6]
    changes = [data[name][-1]["return_change_percent"] for name in ORDER]
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.barh(positions, changes, height=.58, color=COLORS * 2)
    ax.set_yticks(positions, ORDER, fontsize=13, color=INK)
    ax.set_ylim(6.7, -.7)
    ax.set_xlim(-100, 20)
    ax.tick_params(length=0, labelsize=12, colors=MUTED)
    ax.set_xlabel("Mean return change vs. no packet loss (%)", fontsize=13, labelpad=12, color=INK)
    for y, value in zip(positions, changes):
        ax.annotate(f"{value:+.1f}%", (value, y), xytext=(-8 if value < 0 else 8, 0),
                    textcoords="offset points", ha="right" if value < 0 else "left", va="center",
                    fontsize=14, weight="bold", color=INK)
    fig.text(.08, .125, "Battle drops: −20.7% to −83.2%. Pursuit changes: −2.6% to +3.0% in this evaluation.",
             fontsize=16, color=INK, weight="bold")
    save(fig, out, "05_return_change_at_loss50")

    fig, axes = figure("Episode-return variability differs across tasks",
                      "Within-run standard deviation of episode return; these curves are not error bars or between-seed uncertainty.")
    for ax, names, title, limits in zip(axes[0], [battles, pursuits],
                                       ["Battle", "Adversarial Pursuit"], [(0, 2.65), (0, 14)]):
        style(ax, "Episode-return standard deviation", limits, title)
        for name, color in zip(names, COLORS):
            curve(ax, data[name], "return_episode_std", color, name)
        ax.legend(loc="lower left", frameon=False, fontsize=12)
    fig.text(.08, .125, "Battle25/40 reward spread rises at loss 50%; Pursuit has no consistent increase across all maps.",
             fontsize=15, color=INK, weight="bold")
    save(fig, out, "06_episode_return_variability")


def captions(data, audit, out):
    lines = ["# Biểu đồ packet loss — chú thích cho slide", "",
             "## Dữ liệu và cách đọc", "",
             f"Đã đọc {sum(len(item['run_ids']) for item in audit)} JSON W&B: 25 run mới cho 5 map "
             "và 28 bản ghi Battle25 cũ. Battle25 có nhiều bản ghi lặp, nhưng toàn bộ metric test_* "
             "trong mỗi mức loss giống hệt nhau; mỗi mức chỉ được tính là một bộ kết quả. "
             "Tổng cộng 30 điều kiện (6 map × 5 mức loss), mỗi điều kiện 100 episode, config.seed=0 "
             "và env_args.seed=0. Mỗi map dùng một checkpoint đã train; checkpoint khác nhau giữa map. "
             "Không có CI giữa seed và không coi các run lặp là thêm episode độc lập.", "",
             "N là số agent được học, không phải kích thước map. Battle25/35/40 lần lượt có "
             "N=20/42/64; AdvPursuit45/60/70 có N=25/45/61. Tất cả là ExpoComm one-peer.", "",
             "Loss p% là xác suất mất toàn bộ message cho mỗi liên kết từ agent khác ở mỗi lần communication. "
             "Self-link luôn giữ; mất message thì one-peer giữ bộ nhớ message cũ. Đây là loss ở evaluation, "
             "không retrain. Trục x là xác suất cấu hình, không phải tỷ lệ mất đo được từ log.", "",
             "Biểu đồ nối các điểm đã đo để dễ đọc, không chứng minh hiệu năng ở mọi mức nằm giữa. "
             "Các tỉ lệ chuẩn hóa chỉ so với baseline riêng của từng map, không so trực tiếp reward thô giữa hai task.", "",
             "## 01 — Reward được giữ lại so với baseline", "",
             "**Chú thích:** Reward trung bình ở mỗi mức loss được chia cho reward tại loss 0% của cùng map, "
             "rồi nhân 100. Hai panel dùng cùng thang 0–116% để không phóng đại biến động nhỏ của Pursuit.", "",
             "**Thông điệp:** Battle nhạy với mất message hơn AdvPursuit trong các checkpoint được đánh giá. "
             "Ở loss 50%, Battle35 chỉ giữ 16,8% reward baseline, Battle25 giữ 54,4% và Battle40 giữ 79,3%. "
             "AdvPursuit giữ 97,4–103,0%. Không kết luận mô hình cải thiện nhờ loss khi reward nhích tăng.", "",
             "## 02 — Win rate trong Battle", "",
             "**Chú thích:** Tỷ lệ episode mà số agent đỏ sống sót cuối trận lớn hơn số agent xanh, "
             "theo định nghĩa red_team_win trong environment wrapper. Không đồng nghĩa đã tiêu diệt hết đối thủ. "
             "Nếu hai bên sống sót bằng nhau thì không bên nào được tính thắng.", "",
             "**Thông điệp:** Từ loss 0% lên 50%, Battle25 giảm 100% → 67% (−33 điểm phần trăm), "
             "Battle35 giảm 89% → 48% (−41 điểm phần trăm), Battle40 giảm 100% → 98% (−2 điểm phần trăm). "
             "Độ nhạy không tăng đơn điệu theo số agent; không dùng kết quả này để khẳng định map lớn luôn chịu lỗi tốt hơn.", "",
             "## 03 — Reward tuyệt đối cho từng map", "",
             "**Chú thích:** Reward trung bình trên 100 episode ở mỗi mức loss; hàng Battle dùng cùng "
             "thang reward, hàng AdvPursuit dùng cùng thang reward khác. Không so độ cao giữa hai hàng.", "",
             "**Thông điệp:** Battle35 giảm reward từ 1,865 xuống 0,313 tại loss 50%, trong khi "
             "AdvPursuit60 dao động 73,806–76,424. Sự dao động nhỏ hoặc không đơn điệu chưa thể "
             "diễn giải thành cải thiện có ý nghĩa thống kê với một evaluation seed.", "",
             "## 04 — Sống sót và thời lượng trong Battle", "",
             "**Chú thích:** Tỷ lệ sống sót = số agent đỏ sống cuối trận / số agent đỏ ban đầu. "
             "Thời lượng lấy test_episode_length_mean, không phải test_ep_length_mean của runner; "
             "hai metric có thể lệch một bước do cách đếm của wrapper.", "",
             "**Thông điệp:** Tại loss 50%, tỷ lệ agent đỏ còn sống: Battle25 51,0%, Battle35 51,5%, "
             "Battle40 88,0%. Thời lượng Battle25/40 tăng, nhưng Battle35 giảm nhẹ. Vì thời lượng "
             "gồm cả trận thắng lẫn thua, không diễn giải mọi thay đổi thời lượng là tốc độ chiến thắng. "
             "AdvPursuit không có chỉ số sống sót được log; thời lượng 199 bước ở mọi điều kiện nên không thêm đồ thị riêng.", "",
             "## 05 — Tổng kết tác động tại loss 50%", "",
             "**Chú thích:** Phần trăm thay đổi reward so với loss 0% của cùng map. "
             "Giá trị âm là giảm reward; giá trị dương chỉ là chênh lệch quan sát trong lần đánh giá.", "",
             "**Thông điệp:** Mức giảm lớn nhất được quan sát ở Battle35 (−83,2%). "
             "Các thay đổi AdvPursuit nhỏ hơn nhiều: −2,6%, +3,0%, −0,3%. "
             "Chưa có bằng chứng để khẳng định tăng packet loss có lợi cho AdvPursuit60.", "",
             "## 06 — Độ phân tán reward giữa episode", "",
             "**Chú thích:** test_return_std đo độ lệch chuẩn reward giữa các episode trong run. "
             "Đây không phải sai số reward trung bình, không phải SD giữa seed, và không phải CI.", "",
             "**Thông điệp:** Độ phân tán reward ở Battle25/40 tăng rõ tại loss 50%. "
             "Battle35 và AdvPursuit không cho xu hướng tăng nhất quán trên toàn bộ dải loss.", "",
             "## Kết luận có thể dùng trên slide", "",
             "> Trong các checkpoint ExpoComm one-peer được đánh giá với seed 0, packet loss "
             "làm giảm rõ reward và win rate ở Battle, đặc biệt Battle35. AdvPursuit giữ reward "
             "gần baseline trên dải loss 0–50%. Mức ảnh hưởng phụ thuộc map và checkpoint; "
             "cần nhiều evaluation seed và checkpoint train độc lập để khái quát hóa kết quả.", "",
             "## Bảng số liệu chính xác", "",
             "| Map | Loss | Reward | Reward giữ lại (%) | Win rate (%) | Episode reward SD | Episode length |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ORDER:
        for record in data[name]:
            win = "n/a" if record["win_rate"] is None else f"{record['win_rate'] * 100:.0f}"
            lines.append(f"| {name} | {record['loss_percent']}% | {record['return_mean']:.4f} | "
                         f"{record['return_retained_percent']:.2f} | {win} | "
                         f"{record['return_episode_std']:.4f} | {record['episode_length']:.2f} |")
    lines += ["", "PNG 3840×2160 (16:9), SVG vector; số liệu biểu đồ lưu trong plot_data.csv. "
              "source_audit.json liệt kê từng file nguồn và run ID. Script tái tạo: scripts/plot_packet_loss_comparison.py."]
    (out / "CAPTIONS_VI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remaining", type=Path)
    parser.add_argument("--battle25", type=Path, default=Path("results/battle25_packet_loss"))
    parser.add_argument("--out", type=Path, default=Path("output/packet_loss_all_maps"))
    args = parser.parse_args()
    if args.remaining is None:
        candidates = sorted(Path("results").glob("packet_loss_remaining_*"))
        if not candidates:
            parser.error("No packet_loss_remaining export found")
        args.remaining = candidates[-1]
    data, audit = read_exports(args.remaining, args.battle25)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    draw(data, args.out)
    captions(data, audit, args.out)
    records = [record for name in ORDER for record in data[name]]
    # Include nullable survival fraction for Pursuit, without implying a logged value.
    fields = list(dict.fromkeys(key for record in records for key in record))
    with (args.out / "plot_data.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (args.out / "source_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    archive = args.out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(args.out.iterdir()):
            bundle.write(path, args.out.name + "/" + path.name)
    print(f"Read {sum(len(item['run_ids']) for item in audit)} source runs; plotted {len(records)} distinct conditions.")
    for name in ORDER:
        print(f"{name}: return at loss50 changes {data[name][-1]['return_change_percent']:+.2f}%")
    print(f"Saved six 4K PNG/SVG figures, source audit, CSV and Vietnamese captions to {args.out.resolve()}")
    print(f"Archive: {archive.resolve()}")


if __name__ == "__main__":
    main()
