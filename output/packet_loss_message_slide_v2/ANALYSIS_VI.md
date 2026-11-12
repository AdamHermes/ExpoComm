# Speaker notes và phân tích cho slide tiếng Anh đã cập nhật

## Phạm vi của hình

Slide chỉ hiển thị loss 0%, 10%, 20%, 30%, 50%, theo yêu cầu. AdvPursuit dùng Evaluation Return gốc; Battle dùng Evaluation Win Rate (0–1), giống loại metric trong Figure 4 của paper. Đây là các số đã log của nhóm, không phải số paper công bố về packet loss. Bố cục, dữ liệu và biểu đồ native giữ từ slide trước.

Các con số Battle hiện cần gọi là **reported/logged win rate**. Audit đã phát hiện cách đếm agent sống trong wrapper có thể sai ở bước terminal; chưa đối chiếu native MAgent counts trên các episode đã chạy. Không trình bày các số này như winner đã được xác minh độc lập.

## Lời trình bày gợi ý

“Chúng tôi đánh giá độ nhạy của policy đã train khi mất một phần message. Trong dải loss 0–50%, các checkpoint Battle có phản ứng khác nhau. Ví dụ N=42 có logged win rate giảm từ 0,89 xuống 0,48, trong khi N=64 giữ ở mức 0,98. AdvPursuit giữ return gần baseline trên cả ba kích thước đội.

Tuy nhiên, return ổn định không có nghĩa communication không quan trọng. Trong dải đang vẽ, agent vẫn nhận một phần message và có bộ nhớ để tái sử dụng thông tin. Agent cũng có quan sát cục bộ và recurrent state riêng. Do đó kết quả cho thấy độ nhạy với lỗi truyền thông của từng checkpoint, chưa đo toàn bộ lợi ích của communication.

Chúng tôi cần kiểm chứng protocol trước khi claim mạnh hơn. Source code đang reset môi trường hai lần, có thể làm Battle không đổi bên xuất phát giữa các episode. Wrapper cũng đếm quân sống dựa vào key trong observation thay vì native alive counts; cách này có thể đếm cả agent vừa chết. Ngoài ra, các job seed khác tên trước đây vẫn dùng config seed bằng 0. Vì vậy, slide ghi rõ logged metrics và một seed thực tế, không đưa confidence interval giữa seed.

Kết luận ở thời điểm này là: các checkpoint biểu hiện độ nhạy khác nhau với packet loss. Mức đóng góp thực sự của communication cần đối chứng đúng, nhiều seed và baseline train không communication.”

## Kiến thức mới cần lưu ý khi trả lời câu hỏi

1. **Mask loss100 đã qua kiểm tra logic local.** Test tensor với actual ExpoCommOAgent trên sáu kích thước đội cho thấy remote observations không ảnh hưởng Q của agent đang xét tại loss100; đối chứng loss0 có ảnh hưởng. Đây không phải replay các checkpoint trên cluster.
2. **Reset hai lần làm mất luân phiên bên.** _BattleWrapper.reset gọi reset hai lần, còn generate_map đổi leftID/rightID mỗi lần. Probe tái hiện bên khởi tạo giữ cố định qua các episode.
3. **Win-rate implementation cần xác minh native counts.** PettingZoo 1.12.0 lấy observation trước khi loại newly dead agents; key trong observation chưa chắc là agent còn sống ở terminal. Đã tái hiện phản ví dụ bằng method wrapper, nhưng chưa chứng minh winner của từng run W&B thực tế bị đổi.
4. **Battle40 không nhất thiết giữ nguyên chất lượng hành vi.** Logged win rate không giảm dù episode dài hơn và có nhiều quân xanh sống sót hơn. Đây là dữ liệu phụ dùng để chẩn đoán, không thay metric paper trên slide. Win criterion chỉ so số quân hai bên cuối episode. Chưa có action traces để kết luận policy cụ thể đang phòng thủ/camp hay tận dụng side bias.
5. **Return Pursuit là tagging reward.** Return cao trước pretrained prey không chứng minh mọi bài toán cooperative capture đều không cần message. Quan sát cục bộ và phối hợp ngầm qua vị trí/hành động vẫn có thể hỗ trợ policy.

## Loss100 không xuất hiện trong graph nhưng vẫn là bằng chứng cần công bố khi bàn về toàn bộ sweep

Loss100 đã chạy và được audit. Bộ hình này có phạm vi partial loss 0–50%, không đại diện cho toàn bộ sweep. Nếu người nghe hỏi về complete removal, nêu đủ kết quả: logged win rate Battle25/35 bằng 0,00, Battle40 bằng 1,00; return AdvPursuit45/60/70 lần lượt 83,00 / 75,55 / 68,56. Các số vẫn chịu giới hạn protocol và metric vừa nêu. Không suy rộng câu “Pursuit còn nhận message” sang loss100, vì tại đó remote message không bao giờ được nhận trong episode và memory reset ở đầu episode.

## Claim nên dùng

> Within the evaluated checkpoints, reported performance shows different sensitivity to partial message loss. Pursuit returns remain high, while some Battle checkpoints show substantial declines in logged win rate. Stronger claims about communication value require validated outcome counting, balanced starting sides and genuinely independent evaluation seeds.

## Đối chứng tiếp theo

- Log native survivor counts và so old/corrected winner từng episode, cùng end reason.
- Đánh giá cả hai bên xuất phát với reset có kiểm soát. Chạy lại baseline cùng protocol thay vì ghép baseline cũ với protocol mới.
- Xử lý --eval-seed và báo lỗi option không nhận diện; chạy nhiều evaluation seed và checkpoint train độc lập.
- So sánh với baseline train không communication. Một model train có message rồi bị tắt kênh lúc test là một ablation khác.

## Nguồn

- Số liệu graph: output/packet_loss_paper_metrics/reported_values.csv; source_audit.json cùng thư mục.
- Audit mới: output/packet_loss_eval_audit/AUDIT_VI.md và probe_results.json.
- Metric convention: 2502.19717v1.pdf, Figure 4 trang 8.
- Cơ chế: src/modules/agents/ExpoComm_agent.py; src/runners/episode_runner.py; src/envs/battle_wrappers.py; env/battle_v3_view7.py; src/config_overrides.py.
- Terminal ordering: https://raw.githubusercontent.com/Farama-Foundation/PettingZoo/1.12.0/pettingzoo/magent/magent_env.py
