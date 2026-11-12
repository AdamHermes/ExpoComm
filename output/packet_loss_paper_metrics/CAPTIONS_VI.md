# Packet loss — báo cáo đúng metric của paper

Nguồn định nghĩa metric: Figure 4, trang 8, 2502.19717v1.pdf.

- Battle: Evaluation Win Rate, thang 0–1, lấy test_red_team_win_mean.
- AdvPursuit: Evaluation Return gốc, lấy test_return_mean.
- Đây là số đo sweep của nhóm; không phải số lấy từ đường cong trong paper.
- Trục x là packet loss của thí nghiệm bổ sung; Figure 4 gốc dùng training timesteps.
- 100 episode/điều kiện, actual seed 0, một checkpoint/map. Không có CI giữa seed.
- Các bản ghi Battle25 lặp được gộp; không tính thành nhiều seed độc lập.

## Chú thích hình 01

Ảnh hưởng của packet loss tới Evaluation Return trên AdvPursuit và Evaluation Win Rate trên Battle, dùng cùng metric như Figure 4 của paper. Trên các checkpoint này, Battle35 giảm win rate từ 0,89 xuống 0,48, Battle25 từ 1,00 xuống 0,67 và Battle40 từ 1,00 xuống 0,98 khi loss tăng từ 0% lên 50%. Evaluation Return của AdvPursuit45/60/70 ở loss 50% lần lượt là 85,14 / 76,03 / 65,70.

## Chú thích hình 02

Sáu map được trình bày theo bố cục Figure 4: hàng trên là AdvPursuit với Evaluation Return; hàng dưới là Battle với Evaluation Win Rate. Nhãn N là số learning agents, không phải kích thước map.

## Chú thích hình 03

Bảng kết quả trực tiếp từ W&B summary, làm tròn hai chữ số thập phân. reported_values.csv giữ số gốc chưa làm tròn. Không báo cáo normalized return, survival, episode length hay episode reward SD trong bộ hình này.

| Map | Metric | Loss 0% | Loss 10% | Loss 20% | Loss 30% | Loss 50% |
|---|---|---:|---:|---:|---:|---:|
| AdvPursuit45 | Evaluation Return | 87.38 | 87.24 | 87.08 | 85.76 | 85.14 |
| AdvPursuit60 | Evaluation Return | 73.81 | 76.42 | 75.22 | 74.92 | 76.03 |
| AdvPursuit70 | Evaluation Return | 65.87 | 66.49 | 65.36 | 66.51 | 65.70 |
| Battle25 | Evaluation Win Rate | 1.00 | 0.91 | 0.92 | 0.90 | 0.67 |
| Battle35 | Evaluation Win Rate | 0.89 | 0.86 | 0.79 | 0.71 | 0.48 |
| Battle40 | Evaluation Win Rate | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |
