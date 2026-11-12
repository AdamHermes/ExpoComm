# Packet loss 0–100%: kết quả và diễn giải

## Kiểm tra khả năng so sánh

Đã đọc 59 file JSON W&B. Sau khi gộp các bản ghi Battle25 có cùng kết quả, có 36 điều kiện (6 map × 6 mức loss). Mỗi điều kiện khai báo 100 episode, actual seed=0 và env seed=0. Bộ loss 100% có sáu run, một run/map. Toàn bộ logged config của từng run mới khớp config baseline tương ứng, ngoại trừ packet_loss_percent, run_name và work_dir. Summary.timestep cũng khớp. Đây là đối chiếu config đã log, không phải chứng nhận môi trường thực thi hay mã nguồn bit-for-bit giống nhau.

Không có nhiều training/evaluation seed độc lập và không có dữ liệu từng episode trong export. Không vẽ CI giữa seed hoặc coi các bản ghi lặp là thêm episode độc lập.

## Chỉ báo cáo metric trong Figure 4 của paper

Battle: Evaluation Win Rate (0–1), lấy test_red_team_win_mean. AdvPursuit: Evaluation Return gốc, lấy test_return_mean. Không chuẩn hóa reward. N là số learning agents; số trong tên map là kích thước map.

| Map | Metric | Loss 0% | Loss 10% | Loss 20% | Loss 30% | Loss 50% | Loss 100% |
|---|---|---:|---:|---:|---:|---:|---:|
| AdvPursuit45 | Evaluation Return | 87.38 | 87.24 | 87.08 | 85.76 | 85.14 | 83.00 |
| AdvPursuit60 | Evaluation Return | 73.81 | 76.42 | 75.22 | 74.92 | 76.03 | 75.55 |
| AdvPursuit70 | Evaluation Return | 65.87 | 66.49 | 65.36 | 66.51 | 65.70 | 68.56 |
| Battle25 | Evaluation Win Rate | 1.00 | 0.91 | 0.92 | 0.90 | 0.67 | 0.00 |
| Battle35 | Evaluation Win Rate | 0.89 | 0.86 | 0.79 | 0.71 | 0.48 | 0.00 |
| Battle40 | Evaluation Win Rate | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 |

## Battle25 và Battle35: policy phụ thuộc remote-message delivery

Battle25 có win rate 1,00 ở loss 0%, 0,67 ở loss 50%, và 0,00 ở loss 100%. Battle35 lần lượt là 0,89, 0,48 và 0,00. Loss 100% đồng nghĩa hai run đều không thắng trận nào trong 100 episode đã đánh giá. So với loss 50%, win rate giảm lần lượt 0,67 và 0,48 ở đơn vị win rate. Đây là bằng chứng mạnh trong các checkpoint này rằng bỏ hẳn remote-message delivery phá vỡ hiệu năng vốn đạt được khi còn communication.

Các điểm đo chưa cho biết policy giảm mạnh ở loss 60%, 80% hay gần 100%. Vì vậy, đoạn 50–100% được nối nét đứt và không gọi một mức loss chưa đo là ngưỡng sụp đổ.

## Battle40: độ phụ thuộc thay đổi giữa các checkpoint

Win rate Battle40 là 1,00 ở loss 0%, 0,98 ở loss 50% và 1,00 ở loss 100%. Trong metric đang báo cáo, run mới vẫn thắng 100/100 episode khi không nhận remote messages. Do đó không thể kết luận mọi policy Battle đều cần communication để đạt win rate cao. Cũng không thể nói packet loss giúp policy tốt hơn: 1,00 so với 0,98 chỉ là chênh lệch quan sát trong hai tập evaluation hữu hạn.

Win rate trong wrapper là số agent đỏ sống sót cuối episode lớn hơn số agent xanh. Nó không yêu cầu tiêu diệt hết đối thủ và không mô tả đầy đủ mọi khía cạnh chất lượng chiến đấu. Không thay thế win rate bằng reward của Battle trong bộ hình này.

## AdvPursuit: return vẫn cao khi tắt remote delivery

AdvPursuit45 giảm từ 87,3752 xuống 83,0004 khi loss tăng từ 0% lên 100%, chênh −4,3748 đơn vị return. AdvPursuit60 thay đổi từ 73,8057 lên 75,5525 (+1,7468), AdvPursuit70 từ 65,8750 lên 68,5634 (+2,6885). So với loss 50%, các chênh lệch return lần lượt là −2,1368, −0,4802 và +2,8660.

Các checkpoint AdvPursuit vẫn đạt return tương đối cao mà không nhận message từ agent khác. Điều này mở rộng kết luận trước đây: độ ổn định không chỉ do còn một nửa message đến nơi ở loss 50%. Với một seed và không có đối chứng thống kê phù hợp, không khẳng định các mức tăng nhỏ là cải thiện thật, hay message gây hại. Cũng chưa thể kết luận local observation là nguyên nhân duy nhất.

## Điểm quan trọng về bộ nhớ tại loss 100%

Ở loss 50%, agent có thể tái sử dụng thông tin đồng đội đã nhận ở bước trước. Nhưng ở loss 100%, code chặn mọi remote message từ đầu episode, và runner reset hidden/message state ở đầu mỗi episode. Vì vậy không có thông tin đồng đội từ episode trước để giữ lại. Self-link và bộ nhớ xử lý thông tin riêng vẫn hoạt động. Không giải thích sự ổn định của loss 100% bằng việc agent giữ lại remote message từ những episode trước.

## Ý nghĩa thực sự đối với communication

Kết quả cho thấy mức phụ thuộc của policy đã train vào remote-message delivery khác nhau rõ giữa map/checkpoint: thiết yếu để giữ win rate đã quan sát ở Battle25/35, nhưng chưa cần thiết để giữ win rate cao ở Battle40 hay return cao ở ba AdvPursuit trong protocol này. Không được biến kết quả này thành khẳng định universal rằng communication luôn quan trọng hoặc luôn thừa.

Ablation loss 100% tác động đồng thời vào nội dung thông tin nhận được và nhịp cập nhật message GRU (one-peer giữ msg_ego khi thất bại). Nó chưa đo riêng giá trị ngữ nghĩa của nội dung message. Đây cũng là policy đã train cùng communication rồi bị tắt kênh lúc test; không tương đương policy được train từ đầu không communication. Muốn đo lợi ích của phương pháp cần baseline train không communication, nhiều training seed và evaluation seed thực sự khác nhau.

## Chú thích cho slide

**Hình 01:** Evaluation Return của AdvPursuit và Evaluation Win Rate của Battle theo packet loss. Hình thoi đánh dấu loss 100%: không nhận message từ agent khác, self-link vẫn giữ. Nét đứt chỉ nối các điểm đã đo tại 50% và 100%, không mô tả ngưỡng chuyển tiếp.

**Hình 02:** Cùng dữ liệu theo từng map, hàng trên AdvPursuit và hàng dưới Battle. Nhãn số chỉ hiển thị tại loss 0%, 50%, 100% để dễ đọc; CSV giữ đủ sáu mức.

**Hình 03:** So sánh trực tiếp ba chế độ không mất message, mất một phần và chặn toàn bộ remote delivery. Hai panel dùng metric và đơn vị khác nhau theo Figure 4; không so độ cao return với win rate.

**Hình 04:** Bảng số đo gốc, làm tròn hai chữ số. CSV giữ độ chính xác đầy đủ.

**Thông điệp ngắn:** Complete remote-message removal eliminates observed wins in Battle25/35, while Battle40 and the Pursuit checkpoints remain resilient on the reported metrics. Communication dependence is policy- and scenario-specific in this evaluation.

## Nguồn và khả năng tái tạo

Source_audit.json liệt kê toàn bộ run ID và file JSON được dùng. reported_values.csv lưu 36 giá trị metric gốc. Cơ chế cần đối chiếu: src/modules/agents/ExpoComm_agent.py (_packet_keep_mask, ExpoCommOAgent._communicate, init_hidden), src/runners/episode_runner.py (reset/init_hidden), src/envs/battle_wrappers.py (định nghĩa win). Metric convention: Figure 4, trang 8 của 2502.19717v1.pdf. Đây là số đo của nhóm, không phải số paper đã công bố về packet loss.
