# Biểu đồ packet loss — chú thích cho slide

## Dữ liệu và cách đọc

Đã đọc 53 JSON W&B: 25 run mới cho 5 map và 28 bản ghi Battle25 cũ. Battle25 có nhiều bản ghi lặp, nhưng toàn bộ metric test_* trong mỗi mức loss giống hệt nhau; mỗi mức chỉ được tính là một bộ kết quả. Tổng cộng 30 điều kiện (6 map × 5 mức loss), mỗi điều kiện 100 episode, config.seed=0 và env_args.seed=0. Mỗi map dùng một checkpoint đã train; checkpoint khác nhau giữa map. Không có CI giữa seed và không coi các run lặp là thêm episode độc lập.

N là số agent được học, không phải kích thước map. Battle25/35/40 lần lượt có N=20/42/64; AdvPursuit45/60/70 có N=25/45/61. Tất cả là ExpoComm one-peer.

Loss p% là xác suất mất toàn bộ message cho mỗi liên kết từ agent khác ở mỗi lần communication. Self-link luôn giữ; mất message thì one-peer giữ bộ nhớ message cũ. Đây là loss ở evaluation, không retrain. Trục x là xác suất cấu hình, không phải tỷ lệ mất đo được từ log.

Biểu đồ nối các điểm đã đo để dễ đọc, không chứng minh hiệu năng ở mọi mức nằm giữa. Các tỉ lệ chuẩn hóa chỉ so với baseline riêng của từng map, không so trực tiếp reward thô giữa hai task.

## 01 — Reward được giữ lại so với baseline

**Chú thích:** Reward trung bình ở mỗi mức loss được chia cho reward tại loss 0% của cùng map, rồi nhân 100. Hai panel dùng cùng thang 0–116% để không phóng đại biến động nhỏ của Pursuit.

**Thông điệp:** Battle nhạy với mất message hơn AdvPursuit trong các checkpoint được đánh giá. Ở loss 50%, Battle35 chỉ giữ 16,8% reward baseline, Battle25 giữ 54,4% và Battle40 giữ 79,3%. AdvPursuit giữ 97,4–103,0%. Không kết luận mô hình cải thiện nhờ loss khi reward nhích tăng.

## 02 — Win rate trong Battle

**Chú thích:** Tỷ lệ episode mà số agent đỏ sống sót cuối trận lớn hơn số agent xanh, theo định nghĩa red_team_win trong environment wrapper. Không đồng nghĩa đã tiêu diệt hết đối thủ. Nếu hai bên sống sót bằng nhau thì không bên nào được tính thắng.

**Thông điệp:** Từ loss 0% lên 50%, Battle25 giảm 100% → 67% (−33 điểm phần trăm), Battle35 giảm 89% → 48% (−41 điểm phần trăm), Battle40 giảm 100% → 98% (−2 điểm phần trăm). Độ nhạy không tăng đơn điệu theo số agent; không dùng kết quả này để khẳng định map lớn luôn chịu lỗi tốt hơn.

## 03 — Reward tuyệt đối cho từng map

**Chú thích:** Reward trung bình trên 100 episode ở mỗi mức loss; hàng Battle dùng cùng thang reward, hàng AdvPursuit dùng cùng thang reward khác. Không so độ cao giữa hai hàng.

**Thông điệp:** Battle35 giảm reward từ 1,865 xuống 0,313 tại loss 50%, trong khi AdvPursuit60 dao động 73,806–76,424. Sự dao động nhỏ hoặc không đơn điệu chưa thể diễn giải thành cải thiện có ý nghĩa thống kê với một evaluation seed.

## 04 — Sống sót và thời lượng trong Battle

**Chú thích:** Tỷ lệ sống sót = số agent đỏ sống cuối trận / số agent đỏ ban đầu. Thời lượng lấy test_episode_length_mean, không phải test_ep_length_mean của runner; hai metric có thể lệch một bước do cách đếm của wrapper.

**Thông điệp:** Tại loss 50%, tỷ lệ agent đỏ còn sống: Battle25 51,0%, Battle35 51,5%, Battle40 88,0%. Thời lượng Battle25/40 tăng, nhưng Battle35 giảm nhẹ. Vì thời lượng gồm cả trận thắng lẫn thua, không diễn giải mọi thay đổi thời lượng là tốc độ chiến thắng. AdvPursuit không có chỉ số sống sót được log; thời lượng 199 bước ở mọi điều kiện nên không thêm đồ thị riêng.

## 05 — Tổng kết tác động tại loss 50%

**Chú thích:** Phần trăm thay đổi reward so với loss 0% của cùng map. Giá trị âm là giảm reward; giá trị dương chỉ là chênh lệch quan sát trong lần đánh giá.

**Thông điệp:** Mức giảm lớn nhất được quan sát ở Battle35 (−83,2%). Các thay đổi AdvPursuit nhỏ hơn nhiều: −2,6%, +3,0%, −0,3%. Chưa có bằng chứng để khẳng định tăng packet loss có lợi cho AdvPursuit60.

## 06 — Độ phân tán reward giữa episode

**Chú thích:** test_return_std đo độ lệch chuẩn reward giữa các episode trong run. Đây không phải sai số reward trung bình, không phải SD giữa seed, và không phải CI.

**Thông điệp:** Độ phân tán reward ở Battle25/40 tăng rõ tại loss 50%. Battle35 và AdvPursuit không cho xu hướng tăng nhất quán trên toàn bộ dải loss.

## Kết luận có thể dùng trên slide

> Trong các checkpoint ExpoComm one-peer được đánh giá với seed 0, packet loss làm giảm rõ reward và win rate ở Battle, đặc biệt Battle35. AdvPursuit giữ reward gần baseline trên dải loss 0–50%. Mức ảnh hưởng phụ thuộc map và checkpoint; cần nhiều evaluation seed và checkpoint train độc lập để khái quát hóa kết quả.

## Bảng số liệu chính xác

| Map | Loss | Reward | Reward giữ lại (%) | Win rate (%) | Episode reward SD | Episode length |
|---|---:|---:|---:|---:|---:|---:|
| Battle25 | 0% | 4.7343 | 100.00 | 100 | 0.4045 | 100.72 |
| Battle25 | 10% | 3.7289 | 78.76 | 91 | 1.1637 | 148.69 |
| Battle25 | 20% | 3.7930 | 80.12 | 92 | 1.2131 | 141.44 |
| Battle25 | 30% | 3.5614 | 75.23 | 90 | 1.3340 | 153.53 |
| Battle25 | 50% | 2.5777 | 54.45 | 67 | 1.7389 | 167.92 |
| Battle35 | 0% | 1.8648 | 100.00 | 89 | 1.9766 | 199.00 |
| Battle35 | 10% | 1.6462 | 88.27 | 86 | 2.1471 | 195.64 |
| Battle35 | 20% | 1.4046 | 75.32 | 79 | 2.2313 | 198.92 |
| Battle35 | 30% | 1.2756 | 68.40 | 71 | 2.1570 | 194.23 |
| Battle35 | 50% | 0.3132 | 16.79 | 48 | 2.0207 | 191.15 |
| Battle40 | 0% | 4.3689 | 100.00 | 100 | 0.3992 | 83.31 |
| Battle40 | 10% | 4.2230 | 96.66 | 100 | 1.1952 | 90.55 |
| Battle40 | 20% | 4.2724 | 97.79 | 100 | 0.9609 | 87.90 |
| Battle40 | 30% | 4.1498 | 94.98 | 100 | 0.9717 | 93.89 |
| Battle40 | 50% | 3.4655 | 79.32 | 98 | 1.3812 | 120.86 |
| AdvPursuit45 | 0% | 87.3752 | 100.00 | n/a | 11.6893 | 199.00 |
| AdvPursuit45 | 10% | 87.2376 | 99.84 | n/a | 11.2952 | 199.00 |
| AdvPursuit45 | 20% | 87.0823 | 99.66 | n/a | 11.4107 | 199.00 |
| AdvPursuit45 | 30% | 85.7555 | 98.15 | n/a | 9.7614 | 199.00 |
| AdvPursuit45 | 50% | 85.1372 | 97.44 | n/a | 10.1715 | 199.00 |
| AdvPursuit60 | 0% | 73.8057 | 100.00 | n/a | 7.1523 | 199.00 |
| AdvPursuit60 | 10% | 76.4237 | 103.55 | n/a | 8.5203 | 199.00 |
| AdvPursuit60 | 20% | 75.2158 | 101.91 | n/a | 7.4442 | 199.00 |
| AdvPursuit60 | 30% | 74.9227 | 101.51 | n/a | 7.5222 | 199.00 |
| AdvPursuit60 | 50% | 76.0327 | 103.02 | n/a | 7.6281 | 199.00 |
| AdvPursuit70 | 0% | 65.8750 | 100.00 | n/a | 6.8593 | 199.00 |
| AdvPursuit70 | 10% | 66.4898 | 100.93 | n/a | 6.5857 | 199.00 |
| AdvPursuit70 | 20% | 65.3556 | 99.21 | n/a | 6.5482 | 199.00 |
| AdvPursuit70 | 30% | 66.5114 | 100.97 | n/a | 7.2931 | 199.00 |
| AdvPursuit70 | 50% | 65.6974 | 99.73 | n/a | 6.6945 | 199.00 |

PNG 3840×2160 (16:9), SVG vector; số liệu biểu đồ lưu trong plot_data.csv. source_audit.json liệt kê từng file nguồn và run ID. Script tái tạo: scripts/plot_packet_loss_comparison.py.
