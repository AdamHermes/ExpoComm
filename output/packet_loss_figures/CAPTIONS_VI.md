# Packet-loss evaluation: figure captions and slide notes

Các hình có nhãn tiếng Anh, tỷ lệ 16:9. PNG: 3840 × 2160; SVG: vector có thể phóng to.

## Dữ liệu và giới hạn

Đã đọc toàn bộ 28 file JSON của W&B. Các mức 0%, 10%, 20% có 6 bản ghi/mức; 30%, 50% có 5 bản ghi/mức. Toàn bộ config.seed và env_args.seed đều bằng 0. Các metric test_* trong từng mức loss giống hệt nhau. Vì vậy, mỗi mức loss chỉ được biểu diễn bằng một bộ số liệu, không coi các bản ghi lặp là các seed độc lập.

Mỗi run khai báo 100 episode, cùng checkpoint 5.009.240, Battle25, ExpoComm one-peer n6. Không cộng các run lặp thành 500–600 episode độc lập. Không vẽ CI/SD giữa seed. Đường nối chỉ giúp đọc xu hướng giữa các mức đã đo, không phải kết quả tại mọi mức loss.

## 01 — Win rate và reward

**Chú thích dưới hình:** Ảnh hưởng của xác suất mất toàn bộ message trên các liên kết từ agent khác tới hiệu năng ExpoComm. Mỗi điểm biểu diễn kết quả 100 episode của một checkpoint với seed thực tế bằng 0; các run trùng số liệu được gộp để hiển thị.

**Thông điệp khi thuyết trình:** Win rate giữ ở mức 90–92% khi loss từ 10% đến 30%, nhưng giảm còn 67% tại loss 50%. So với không mất message, win rate giảm 33 điểm phần trăm và reward giảm 45.6%. Chưa đủ bằng chứng để kết luận 20% tốt hơn 10% hoặc xác định một ngưỡng suy giảm chính xác.

## 02 — Agent sống sót và thời lượng episode

**Chú thích dưới hình:** Số agent đỏ sống sót trung bình và số bước trung bình mỗi episode theo mức packet loss. Thời lượng sử dụng metric test_episode_length_mean của môi trường.

**Thông điệp khi thuyết trình:** Khi loss tăng từ 0% lên 50%, số agent sống sót giảm từ 17,22 xuống 10,20 (−40.8%), trong khi thời lượng episode tăng từ 100,72 lên 167,92 bước (+66.7%). Kết hợp với reward/win rate giảm, kết quả phù hợp với sự suy giảm hiệu quả phối hợp. Đây là diễn giải từ các metric, chưa phải phép đo trực tiếp chất lượng phối hợp hay thời gian thắng; thời lượng trung bình bao gồm cả trận thắng và trận thua.

## 03 — Độ phân tán reward giữa episode

**Chú thích dưới hình:** Độ lệch chuẩn reward giữa các episode trong từng run (test_return_std). Chỉ số này đo mức phân tán kết quả trận đấu, không phải sai số của reward trung bình hay độ lệch chuẩn giữa các seed.

**Thông điệp khi thuyết trình:** Độ lệch chuẩn reward tăng từ 0,405 lên 1,739, tương đương 4.30 lần, khi loss tăng từ 0% lên 50%. Các trận có kết quả phân tán hơn trong cấu hình đánh giá này.

## Bảng số liệu chính xác

| Loss | Win rate (%) | Reward mean | Episode reward SD | Red survivors | Episode length |
|---|---:|---:|---:|---:|---:|
| 0% | 100 | 4.7343 | 0.4045 | 17.22 | 100.72 |
| 10% | 91 | 3.7289 | 1.1637 | 15.22 | 148.69 |
| 20% | 92 | 3.7930 | 1.2131 | 15.22 | 141.44 |
| 30% | 90 | 3.5614 | 1.3340 | 14.75 | 153.53 |
| 50% | 67 | 2.5777 | 1.7389 | 10.20 | 167.92 |

**Nguồn:** JSON W&B export trong results/battle25_packet_loss/runs. source_audit.json lưu danh sách run và đường dẫn nguồn; plot_data.csv lưu số liệu được vẽ.
