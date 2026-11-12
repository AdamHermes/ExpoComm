# Vai trò thực sự của message trong ExpoComm

## Lời trình bày gợi ý cho một slide

“Chúng tôi dùng hai metric giống Figure 4 của paper: Evaluation Return cho AdvPursuit và Evaluation Win Rate cho Battle. Mỗi map dùng cùng một checkpoint cho toàn bộ mức loss.

Ở Battle, mất message làm giảm hiệu năng, nhưng mức giảm phụ thuộc map và checkpoint. Khi tăng loss từ 0% lên 50%, win rate Battle35, tương ứng N=42, giảm từ 0,89 xuống 0,48. Battle25 giảm từ 1,00 xuống 0,67, trong khi Battle40 chỉ giảm từ 1,00 xuống 0,98. Điều này cho thấy việc nhận được các message cập nhật có ảnh hưởng tới hành vi và kết quả của policy đã học, đặc biệt ở Battle35.

AdvPursuit giữ Evaluation Return khá gần baseline. Tuy nhiên, tại loss 50%, agent vẫn có xác suất nhận mỗi message từ agent khác là 50%. Khi mất message, agent tiếp tục giữ bộ nhớ communication cũ; self-link cũng luôn được giữ. Vì vậy, agent vẫn có thể sử dụng thông tin đồng đội đã nhận trước đó. Sự ổn định trước mất một phần message chưa chứng minh rằng communication không có ích.

Kết luận phù hợp là: các checkpoint có độ nhạy khác nhau với việc mất thông tin mới. Để đo đầy đủ lợi ích của message, cần thêm điều kiện tắt hoàn toàn remote communication và baseline train không có communication, với nhiều seed.”

## Số liệu được báo cáo

| Map | Số learning agents N | Metric của paper | Loss 0% | Loss 50% |
|---|---:|---|---:|---:|
| Battle25 | 20 | Evaluation Win Rate | 1,00 | 0,67 |
| Battle35 | 42 | Evaluation Win Rate | 0,89 | 0,48 |
| Battle40 | 64 | Evaluation Win Rate | 1,00 | 0,98 |
| AdvPursuit45 | 25 | Evaluation Return | 87,38 | 85,14 |
| AdvPursuit60 | 45 | Evaluation Return | 73,81 | 76,03 |
| AdvPursuit70 | 61 | Evaluation Return | 65,87 | 65,70 |

Slide không sử dụng reward chuẩn hóa. N là số agent được học, không phải kích thước map.

## Message có thể đóng góp gì?

Agent chọn action từ quan sát cục bộ, hidden state riêng và bộ nhớ message. Message có thể mang thông tin từ lịch sử hoặc quan sát của agent khác mà agent nhận chưa trực tiếp thấy. Về cơ chế, đây là một kênh bổ sung thông tin cho quyết định phân tán. Các tensor message là biểu diễn được học; kết quả sweep không cho phép gán chắc chắn một ý nghĩa cụ thể như “tọa độ mục tiêu” hoặc “kế hoạch tấn công” cho chúng.

Sự giảm win rate ở Battle khi làm gián đoạn kênh message là bằng chứng về độ phụ thuộc của các policy này vào việc truyền thông tin. Nó chưa đo trực tiếp message chứa bao nhiêu thông tin hữu ích hoặc chiến thuật cụ thể nào bị mất.

## Sweep hiện tại đo điều gì?

Code lấy mẫu một mask Bernoulli cho từng liên kết được chọn ở mỗi lần communication. Loss là xác suất bỏ toàn bộ message từ agent khác, không phải mask từng phần tử tensor. Self-link không mất. Trong one-peer, nếu delivery thất bại thì nhánh communication giữ msg_ego thay vì dùng candidate_msg của GRU. Hidden state xử lý quan sát riêng vẫn cập nhật.

Do đó, thí nghiệm làm gián đoạn luồng cập nhật communication và cả nhịp cập nhật bộ nhớ. Nó đánh giá độ nhạy của một policy cố định đối với cơ chế lỗi này, chưa tách riêng ảnh hưởng của nội dung message khỏi ảnh hưởng của nhịp cập nhật và phân phối đầu vào.

## Vì sao Pursuit có thể ổn định?

Một số giả thuyết phù hợp với cơ chế là: bộ nhớ giữ thông tin đủ lâu, thông tin mới trùng lặp qua nhiều lượt truyền, hoặc quan sát cục bộ đã đủ hữu ích cho các checkpoint này. Dữ liệu hiện tại không phân biệt được các giả thuyết đó. Loss 50% cũng không tương đương tắt message ở 50% timestep toàn hệ thống: các remote link lấy mẫu độc lập và các lượt self-link luôn thành công.

Không nên diễn giải reward AdvPursuit60 tăng nhẹ là lợi ích chắc chắn của packet loss. Cũng không suy ra số agent lớn hơn thì ít cần communication hơn, vì map, checkpoint và policy đã học đều thay đổi giữa các cấu hình.

## Thí nghiệm tiếp theo để đo đầy đủ lợi ích communication

1. **Tắt toàn bộ remote communication lúc evaluation:** đặt loss 100% cho các liên kết từ agent khác, giữ self-link và reset bộ nhớ như bình thường ở đầu episode. Đây là ablation về độ phụ thuộc vào remote communication của checkpoint đã train, không tự động tương đương một policy được train từ đầu mà không có communication.
2. **Train baseline không communication:** dùng cùng tài nguyên train, task, network capacity phù hợp và nhiều training seed để so sánh trực tiếp lợi ích đạt được nhờ communication.
3. **Kiểm soát nội dung và bộ nhớ:** so sánh giữ message cũ, thay bằng zero, và shuffle message giữa các agent trong khi kiểm soát cơ chế cập nhật. Thiết kế đối chứng cần tính tới distribution shift; các phép thay thế không hoàn toàn tương đương nhau.
4. **Kiểm tra lỗi liên tiếp hoặc độ trễ:** so sánh với loss độc lập hiện tại để xem bộ nhớ chịu được khoảng thiếu thông tin mới dài bao lâu.

Ưu tiên thực tế là thêm loss 100% và nhiều evaluation seed được áp dụng thực sự, trước khi khái quát về tầm quan trọng của message. Muốn kết luận về ưu thế của phương pháp, cần nhiều checkpoint train độc lập và baseline phù hợp.

## Giới hạn và nguồn

- Mỗi điều kiện khai báo 100 episode, actual config.seed=0 và env_args.seed=0. Mỗi map dùng một checkpoint đã train. Không có CI giữa seed. Các run Battle25 lặp cho cùng metric đã được gộp, không tính là các thí nghiệm độc lập.
- Win rate Battle theo wrapper là số agent đỏ sống sót cuối episode lớn hơn số agent xanh, không yêu cầu tiêu diệt hết đối thủ.
- Nguồn số liệu: output/packet_loss_paper_metrics/reported_values.csv và các JSON được liệt kê trong output/packet_loss_paper_metrics/source_audit.json. Dữ liệu mới trong results/packet_loss_remaining_20261110_141557_124749; Battle25 từ results/battle25_packet_loss.
- Nguồn định nghĩa metric: 2502.19717v1.pdf, Figure 4, trang 8. Đây là metric convention của paper, còn số liệu packet loss là kết quả nhóm đã chạy. Trục ngang Figure 4 gốc là training timestep; trục ngang slide này là packet loss.
- Nguồn cơ chế: src/modules/agents/ExpoComm_agent.py, _packet_keep_mask và ExpoCommOAgent._communicate; src/controllers/ExpoComm_controller.py; src/envs/battle_wrappers.py.
