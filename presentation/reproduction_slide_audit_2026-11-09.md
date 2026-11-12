# Đối chiếu paper, kết quả trong repo và slide ExpoComm

Ngày kiểm tra: 09/11/2026. Phạm vi: nội dung và số liệu của `ExpoComm_Group33_native.pptx` (17 slide), đối chiếu `2502.19717v1.pdf` và dữ liệu hiện có trong bản checkout này. Không sửa slide.

## Nguồn kết quả của nhóm dùng để kiểm tra slide

Theo xác nhận của nhóm, nguồn số liệu chính là các lần chạy của nhóm:

| Kết quả | Nguồn trong repo |
|---|---|
| Static Pursuit | `presentation/pursuit/seed1/` |
| Static Battle | `wandb_exports/ExpoComm_Battle_static/` |
| One-peer Pursuit | `wandb_exports/ExpoComm_AdvPursuit45/7oohgveb/`, `ExpoComm_AdvPursuit60/lky5a2xa/`, `ExpoComm_AdvPursuit70/8vis0wl2/` |
| One-peer Battle | `wandb_exports/ExpoComm_Battle25/tdltdzof/`, `ExpoComm_Battle35/hjv89i01/`, `ExpoComm_Battle40/68wgbec8/` |
| IMP và Battle transfer | `presentation/page9/imp_table1/` và `presentation/page9/battle_fig5/` |

Paper dùng để hiểu technical details và đọc các so sánh có gắn nhãn “paper”; không dùng số liệu paper thay cho kết quả của nhóm.

Đã mở và kiểm tra hai ảnh Static Battle: `battle_static_test_red_team_win_mean.png` và `battle_static_test_return_mean.png`. Đây là kết quả của nhóm. Ba đường static Battle trên slide 13 tương ứng về hình dạng với ảnh win-rate này. Giới hạn xác minh là nguồn lưu dưới dạng ảnh, nên giá trị trace trong chart là xấp xỉ; không có nghĩa là thiếu kết quả Static Battle hoặc slide lấy các đường này từ paper. Ảnh return Static Battle chưa được trình bày trong deck; có thể thêm appendix nếu muốn bao phủ cả metric phụ này.

## Kết luận

Slide đã bao phủ các kết quả chính mà nhóm nêu: static và one-peer trên cả Pursuit và Battle, cùng đủ 30 kết quả IMP. Kết quả chính được đối chiếu với các nguồn của nhóm ở bảng trên. Phần lớn số liệu kiểm tra được là đúng. Cần làm rõ protocol thực tế, độ chính xác khi trace ảnh Static Battle, hạn chế so sánh khác seed, và phạm vi baseline trong transfer. Đây là đánh giá độ đúng/đủ của báo cáo kết quả nhóm; việc xác nhận lại toàn bộ kết luận paper là một mức yêu cầu khác.

Phân biệt ba mức xác minh:

- **Đối chiếu được với chuỗi đánh giá:** 3 đường static Pursuit và 6 đường one-peer MAgent.
- **Đối chiếu được với dữ liệu tổng hợp đã lưu:** 30 ô IMP, 9 cấu hình transfer, các thống kê hành vi.
- **Chưa đối chiếu được với log gốc:** 3 đường static Battle; quá trình training của 30 run IMP; checkpoint/config dùng cho transfer. Các đường dẫn nguồn có trong dữ liệu nhưng nhiều đường dẫn không tồn tại trong checkout này.

## Các technical details trong paper

Nguồn: mục 3.1–3.3, mục 4, Appendix B và C của PDF.

- Static exponential dùng các khoảng cách là lũy thừa của 2. Budget thí nghiệm được paper gọi là K = ceil(log2 N), có tổng số liên kết ngoài agent theo bậc O(N log N).
- One-peer luân phiên các khoảng cách exponential, với budget danh nghĩa K = 1 và O(N) liên kết mỗi bước.
- Khả năng lan truyền thông tin trong O(log N) bước là tính chất topology với điều kiện thông tin còn hữu dụng và được giữ lại khi tổng hợp. Điều này không bảo đảm learned message luôn giữ thông tin không mất mát.
- Static tổng hợp message bằng attention; one-peer dùng GRU. Cả hai kết hợp local history và message để chọn action.
- MAgent: IDQN/IQL với individual rewards, auxiliary loss dự đoán global state. Pursuit có 25/45/61 learning agents, map 45/60/70. Battle có 20/42/64 learning agents, map 25/35/40. Đối thủ benchmark là pretrained policies.
- IMP: QMIX với shared reward, campaign cost; Uncorrelated/Correlated/OWF, N = 50/100. Auxiliary loss là InfoNCE. IMP có global state là concatenation của observations nhưng paper xem biểu diễn này không compact/phù hợp cho prediction grounding; không nên diễn giải thành IMP hoàn toàn không có state.
- Paper dùng 5 training seeds. Figure 4 có 95% CI; Table 1 dùng mean và SD qua seeds. Mỗi evaluation có 100 episodes.
- Paper: 5 triệu bước cho MAgent, 2 triệu cho IMP; test interval tương ứng 50.000 và 25.000 bước. Table 2 ghi replay buffer 2.000.
- Transfer: policies đấu trực tiếp với nhau; mỗi cặp đấu 200 games; đếm số đối thủ mà mỗi policy thắng nhiều game hơn. Đây là win count theo matchup, khác win rate trước pretrained opponent trong Figure 4.

## Kết quả xác minh theo slide

| Slide | Đánh giá | Chi tiết / chỉnh sửa đề nghị |
|---|---|---|
| 6–7 | Đúng ở mức giải thích topology | Ví dụ N=8, +1/+2/+4 và điều kiện retained, lossless messages là hợp lý. Cần ghi chú thêm khác biệt self-slot của code nếu dùng để giải thích chính xác run thực tế. |
| 8 | Đúng về one-peer và auxiliary loss | Nên thêm một dòng: static dùng attention. Khi nói IMP dùng contrastive, nêu state không compact cho prediction grounding. |
| 9–10 | Cơ bản đúng về paper | Nên ghi rõ đây là benchmark và baseline của paper. ER trong paper lấy mẫu K-in-regular directed graphs, không chỉ một random graph bất kỳ. |
| 11 | Dễ bị hiểu thành protocol của nhóm | Đổi tiêu đề thành “Paper evaluation protocol”, hoặc chia rõ paper / ours. Câu “5 seeds, 95% CI” không áp dụng cho reproduction một seed. Hardware và thời gian ở đây là mô tả của paper, chưa xác minh cho từng run của nhóm. |
| 12 | Bao phủ đúng phần đã chạy, nhưng cần sửa transfer | Đổi “same: 18,000 games” thành “5 policy configurations, 200 games/pair, 18,000 games total”. Paper có DGN+TarMAC trong Figure 5; nhóm không có baseline này. |
| 13 | Có đủ 12 đường ExpoComm, đúng 9 đường có dữ liệu kiểm chứng | Claim 5/6 phù hợp với các giá trị đang vẽ; đây là quan sát giữa các run cụ thể, chưa phải kết luận thống kê. Thêm seed, run ID, cách nội suy, và nguồn static Battle. |
| 14 | Số liệu đúng | Tag success 96,6% là rewarded-tag-events/tag-attempts suy ra từ một probe episode; không phải tỷ lệ bắt được unique prey hay mean qua 100 episodes. Ghi rõ n=1 probe và initial checkpoint 200 bước. Battle 1,00; survivors 57,75 vs 1,10 khớp W&B summary. |
| 15 | Đúng và đủ 30 ô kết quả IMP | Chuẩn hóa, làm tròn và thứ hạng đều đúng. “Static best 4/6, second 2/6” đúng. Có raw return cuối kém không đủ để kết luận run đã “collapsed in training”. |
| 16 | Bảng đếm và phần trăm khớp JSON | Đổi “Left side wins” thành “Red-side wins”: JSON lưu red_side_wins. Bổ sung epsilon=0,05, số matchup hòa, và subset baseline. |
| 17 | Cần thu hẹp kết luận | “Core claim holds on MAgent” rộng hơn bằng chứng hiện có: chưa có baseline learning curves MAgent đầy đủ và chỉ một seed/configuration. “IMP and transfer gains are small” cũng quá gộp: IMP có các chênh lệch lớn và outlier, transfer bị side bias mạnh. |

## MAgent: các đường cong và endpoint

Các chart native trong slide 13 lưu 101 điểm trên lưới 0–5 triệu bước, cách nhau 50.000 bước. Sáu đường one-peer và ba đường static Pursuit khớp nội suy tuyến tính từ nguồn tương ứng; sai số tối đa nhỏ hơn 0,00054 (ở đơn vị metric của từng chart). Sai số nhỏ này phù hợp với việc làm tròn/nội suy khi lưu chart.

Do đó, nhãn “raw” hiện tại chưa chính xác hoàn toàn: nên dùng “unsmoothed, resampled”; đường đậm được slide mô tả là moving average. Chưa xác minh được đầy đủ thứ tự smoothing/nội suy và xử lý đầu/cuối chuỗi vì không có script dựng chart. Ba đường static Battle được slide tự ghi là trace từ W&B plot export, vì vậy cần gọi rõ là digitized estimates và không trình bày như raw evaluation logs.

### Giá trị chưa smoothing mà slide vẽ tại đúng 5 triệu bước

| Scenario | N | Static | One-peer | So sánh trên chart |
|---|---:|---:|---:|---|
| Pursuit | 25 | 69,0799 | 84,5737 | One-peer cao hơn |
| Pursuit | 45 | 61,2704 | 75,3243 | One-peer cao hơn |
| Pursuit | 61 | 69,1617 | 66,1880 | Static cao hơn |
| Battle | 20 | 0,2553* | 1,0000 | One-peer cao hơn |
| Battle | 42 | 0,4998* | 0,8660 | One-peer cao hơn |
| Battle | 64 | 0,9955* | 1,0000 | Gần bằng nhau |

\* Static Battle là dữ liệu đã trace trong deck, chưa kiểm chứng được từ log gốc trong repo này. Chênh lệch 0,9955 vs 1,0 không chứng minh khác biệt thực chất; trace có sai số.

### Evaluation cuối cùng thực sự được lưu

| Scenario | N | Static, seed 1 | One-peer, seed 0 | Ghi chú |
|---|---:|---:|---:|---|
| Pursuit | 25 | 69,0828 ± 11,9137 | 86,6662 | Static tại 5.000.200; one-peer tại 5.050.200 bước |
| Pursuit | 45 | 61,2777 ± 7,1840 | 74,1747 | Cùng quy ước bước như dòng trên |
| Pursuit | 61 | 69,1652 ± 6,1390 | 67,3900 | Cùng quy ước bước như dòng trên |
| Battle | 20 | Chưa có log gốc | 1,00 | One-peer evaluation cuối tại 5.009.240 bước |
| Battle | 42 | Chưa có log gốc | 0,85 | One-peer evaluation cuối tại 5.008.875 bước |
| Battle | 64 | Chưa có log gốc | 1,00 | One-peer evaluation cuối tại 5.008.330 bước |

Các dấu ± ở static Pursuit là SD giữa 100 evaluation episodes, không phải SD giữa training seeds. Khác biệt giữa endpoint trong chart và evaluation cuối là do chart nội suy ở mốc 5 triệu bước; không phải tự thân là lỗi số liệu. Slide nên công bố rõ quy ước này.

Các run one-peer có seed 0 trong `wandb_exports/*/*/config.json`. Static Pursuit có seed 1 trong CSV và manifest. Bởi vậy “one seed per curve” đúng nhưng chưa đủ: người xem cần biết hai variant không phải matched-seed comparison.

Các config W&B one-peer MAgent có replay buffer 5.000, trong khi paper Table 2 ghi 2.000. Không được mô tả reproduction là giữ nguyên toàn bộ hyperparameters chỉ dựa trên tên method. Manifest static Pursuit có provenance và hash cho merged config, nhưng chính các merged config đó không có trong checkout để kiểm tra trực tiếp.

### Chi tiết budget trong code hiện tại

`src/controllers/ExpoComm_controller.py` tạo danh sách `[self, +1, +2, ...]`. `topk_neighbors` bao gồm self. Static dùng toàn bộ danh sách; one-peer chọn `t % topk_neighbors`, nên có một bước chỉ chọn self trong mỗi chu kỳ.

- `topk_neighbors=6`: 5 external peers cho static, one-peer có 1 self-slot/6 bước.
- `topk_neighbors=7`: 6 external peers cho static, one-peer có 1 self-slot/7 bước.

Vì vậy cần phân biệt budget danh nghĩa của paper, slot trong implementation và external network traffic. Các bậc O(N), O(N log N) vẫn đúng. Câu “một external peer mỗi timestep” là mô hình lý tưởng giản lược, chưa mô tả đủ code hiện tại. Chưa có frozen source của từng run trong checkout để khẳng định mọi run đã lưu đều chạy đúng revision này.

## IMP: số liệu đầy đủ và thứ hạng đúng

Nguồn: `presentation/page9/imp_table1/table1.csv`, 30 dòng, mỗi configuration một seed 0.

Công thức kiểm tra: score (%) = 100 × (x − H) / |H|, với x là discounted return. Dùng đúng H từ paper Table 5:

| Scenario | H, N=50 | H, N=100 |
|---|---:|---:|
| Uncorrelated | -232,7 | -231,5 |
| Correlated | -211,0 | -194,0 |
| OWF | -1248,2 | -2436,3 |

Cả 30 giá trị chuẩn hóa đều khớp raw_return trong CSV; các giá trị paper nhỏ trong slide cũng khớp Table 1 sau làm tròn.

| Scenario | N | QMIX | ER K=1 | ER static | ExpoComm K=1 | ExpoComm static |
|---|---:|---:|---:|---:|---:|---:|
| Uncorrelated | 50 | 24,89 | 28,15 | **31,46** | 26,94 | 28,62 |
| Correlated | 50 | -27,95 | 44,87 | 44,01 | 43,43 | **47,88** |
| OWF | 50 | 63,12 | 60,90 | 64,91 | 63,78 | **65,03** |
| Uncorrelated | 100 | 22,92 | 22,31 | -699,06 | 18,99 | **24,77** |
| Correlated | 100 | 34,20 | 33,62 | 23,07 | 29,35 | **34,76** |
| OWF | 100 | 64,75 | 64,01 | **66,66** | 64,00 | 66,35 |

Chữ đậm chỉ giá trị tốt nhất mỗi scenario. Thứ hạng tổng thể: static ExpoComm best 4/6, second 2/6.

Các hạn chế cần ghi:

- 30 đường dẫn `file` tới `stats.jsonl` đều không tồn tại trong checkout. Xác minh được bảng tổng hợp và phép tính, chưa xác minh được diễn tiến training hoặc số episodes thực tế từng run.
- ER static Uncorrelated100 có raw return -1849,8197, tương ứng -699,0582%. Đây là một kết quả cuối rất kém, nhưng CSV không chứng minh được nguyên nhân hay việc “collapse trong training”. Nên dùng “very poor final evaluation” nếu chưa bổ sung trajectory.
- Nhiều chênh lệch static ExpoComm với đối thủ sát nhất rất nhỏ: OWF50 khoảng 0,117 điểm %, Correlated100 khoảng 0,558 điểm %. Không có repeated seeds để kết luận superiority ổn định.
- One-peer thua static trên cả 6 scenario IMP trong bộ dữ liệu này. Đây là khác biệt đáng nói so với paper; slide hiện đã nêu đúng hướng khác biệt này.

## Zero-shot transfer

Nguồn: 9 JSON trong `presentation/page9/battle_fig5/`.

- Mỗi JSON có 5 policy configurations, 10 unordered pairs, 200 games/pair = 2.000 games.
- Tổng: 9 settings × 2.000 = 18.000 games, 90 matchups.
- Win count cộng lại: ExpoComm one-peer 16; ExpoComm static 13; ER one-peer 13; ER static 22; IDQN 13. Tổng có 77 decisive matchups và 13 matchups hòa về số wins. Hòa matchup khác hòa game.
- Các ô đếm trong slide 16 khớp JSON. Các tỷ lệ red-side wins cũng khớp sau làm tròn: 72%, 79%, 95%, 96%, 76%, 91%, 96%, 74%, 81% theo thứ tự slide.
- JSON ghi `eval_epsilon=0.05`; khác greedy evaluation epsilon=0 trong các config one-peer benchmark. Paper không mô tả rõ epsilon transfer ở đoạn protocol, nên chưa kết luận là sai paper; vẫn cần công bố setting này.
- Paper Figure 5 có DGN+TarMAC với cả hai budget; bộ transfer này không có. Không so trực tiếp win-count tuyệt đối giữa hai bảng như thể cùng tập đối thủ.
- JSON trỏ tới checkpoint của những run Battle tháng 11, trong khi one-peer learning curves W&B là run tháng 10. Do đó, cần ghi rõ đây là bộ checkpoint khác; không mặc định transfer dùng chính run trên slide 13. Các checkpoint được trỏ tới không có trong checkout hiện tại.
- `red_side_wins` chứng minh red-side bias rất mạnh. JSON tự nó không xác định red luôn hiển thị ở bên trái, cũng không chứng minh nguyên nhân bias. Nên dùng đúng tên red side, tránh quy nguyên nhân thành lợi thế vị trí xuất phát nếu chưa kiểm tra evaluator/spawn logic.

## Cách sửa phần kết luận mà không cần chạy thêm

Đề nghị thay phần cuối slide 17 bằng:

> In our single-seed runs, one-peer ExpoComm matches or exceeds static ExpoComm in 5 of 6 MAgent settings, using a lower nominal communication budget. Static ExpoComm ranks first in 4 of 6 IMP settings and second in the other two. These runs provide partial support for the paper's findings; MAgent baseline superiority and robust zero-shot transfer remain unverified.

Các bổ sung có thể dùng dữ liệu sẵn có:

1. Một bảng appendix ghi N, map size, variant, seed, run ID, checkpoint/evaluation step, final metric và nguồn; tách trace khỏi log gốc.
2. Bảng final Pursuit kèm episode SD, ghi rõ đây là 100 episodes từ một training seed.
3. Footnote slide 13 về nội suy và moving average; chỉ rõ các giá trị paper là đọc xấp xỉ từ Figure 4.
4. Footnote slide 14 về một probe episode, inferred rewarded tags, và checkpoint ban đầu 200 bước.
5. Footnote slide 16 về epsilon, 5 policy configurations, 13 tied matchups và red-side bias.
6. Lưu lại raw static Battle logs, IMP stats/configs, transfer evaluator/configs/checkpoint manifest để người khác kiểm chứng đầy đủ.

Không cần thêm run chỉ để sửa cách báo cáo hiện tại. Nếu muốn bảo vệ kết luận về tính ổn định, superiority trước baseline hoặc nguyên nhân memory/auxiliary loss, cần thêm matched-seed runs, baseline comparisons và ablation tương ứng.

## Giới hạn kiểm tra

Đã đọc PDF bằng text extraction và kiểm tra trực quan các trang Figure 4/Table 1/Figure 5. Đã đọc toàn bộ text, notes và dữ liệu cache của 6 native charts trong PPTX. Các notes hiện chỉ có số trang, chưa chứa citation/provenance thí nghiệm.

Không kiểm tra được toàn bộ layout trong PowerPoint: COM automation thất bại với lỗi logon session 0x80070520. Vì vậy báo cáo này kết luận về nội dung và số liệu, không xác nhận slide không có lỗi tràn chữ hoặc layout khi trình chiếu.
