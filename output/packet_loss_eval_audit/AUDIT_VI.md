# Audit evaluation packet loss và trường hợp Battle40

## Kết luận

Chưa tìm thấy lỗi khiến mask 100% cho message từ agent khác lọt qua trong đường one-peer đã kiểm tra. Tuy nhiên, phát hiện các vấn đề thực sự trong protocol/metric: CLI bỏ qua eval seed, reset hai lần làm mất luân phiên bên của Battle, và cách đếm quân sống bằng membership của observation có thể đếm cả agent vừa chết ở bước cuối. Vì thế, cần gọi các đồ thị hiện có là **logged evaluation metrics** và kiểm chứng lại win rate trước khi khái quát hóa.

Không sửa source evaluation trong lần audit này. Các kết quả cũ vẫn phản ánh đúng số đã log, nhưng không tự động chứng minh metric/episode sampling là đúng.

## 1. Mask 100%: kiểm chứng được gì?

Đã chạy scripts/audit_packet_loss_eval.py với chính ExpoCommOAgent của repo, sử dụng trọng số ngẫu nhiên và tensor tổng hợp. Bao phủ N=20/42/64 của Battle và N=25/45/61 của AdvPursuit.

- Tất cả remote keep-mask bằng False tại loss 100%; self-link vẫn True.
- Trong rollout nhiều chu kỳ communication, thay đổi toàn bộ quan sát của các agent khác không thay đổi Q-values của agent 0 ở loss 100%.
- Đối chứng loss 0% với cùng trọng số có thay đổi Q-values khi thay quan sát agent khác, xác nhận test thực sự kích hoạt communication.
- init_hidden trả về hidden/message bằng 0; EpisodeRunner gọi init_hidden ở đầu mỗi episode.
- Controller truyền test_mode xuống agent; CLI --packet-loss-percent=100 và --packet-loss-mode=test có áp dụng.

Đây là kiểm chứng logic đường truyền bằng code local, không phải replay các checkpoint đã chạy trên cluster và không phải chứng nhận binary/dependency trên cluster giống hệt máy local. Các checkpoint thực tế tương ứng không có trong checkout local để mô phỏng lại chiến thuật.

## 2. Lỗi xác định: tham số seed bị bỏ qua

src/config_overrides.py chưa xử lý --eval-seed và --eval-result-path. Sau khi gọi apply_packet_loss_cli_overrides với --eval-seed=4, config.seed vẫn bằng 0 và hai option còn nguyên trong params. main.py không báo lỗi với params còn lại. Vì vậy, các job được đặt tên seed1..seed4 trước đây không phải các seed khác nhau. W&B config đã xác nhận toàn bộ seed thực tế là 0.

Các job 100% và đợt sáu map gần đây đã chủ động dùng seed 0, nên vấn đề này không có nghĩa loss 100% bị bỏ qua. Nó vô hiệu hóa cách diễn giải về nhiều seed và local JSON export từ những option chưa được xử lý.

## 3. Lỗi xác định: reset hai lần triệt tiêu đổi bên trong Battle

_BattleWrapper.reset và _AdvPursuitWrapper.reset đều gọi self.env.reset hai lần liên tiếp. Trong env/battle_v3_view7.py, generate_map đổi leftID/rightID mỗi lần. PettingZoo 1.12.0 reset gọi generate_map, nên reset hai lần quay lại cùng bên cũ.

Probe dùng chính method generate_map và _BattleWrapper.reset của repo cho kết quả left-team-ID [0,0,0,0] qua bốn episode, trong khi reset một lần cho [1,0,1,0]. Hướng cố định cụ thể trong simulator còn phụ thuộc reset khởi tạo, nhưng việc không đổi bên giữa episode không phụ thuộc hướng ban đầu.

Hệ quả: episode không cân bằng hai bên như ý định trong generate_map. Điều này có thể tương tác với policy/đối thủ và bố trí bản đồ, đặc biệt nếu policy học hành vi gắn với hướng. Audit chưa chứng minh đây là nguyên nhân duy nhất của Battle40. Cần eval cả hai bên bằng protocol có kiểm soát; nếu sửa reset, phải chạy lại baseline và loss100 cùng protocol mới.

## 4. Lỗi metric có thể ảnh hưởng winner: observation membership không phải true alive

_BattleWrapper.step tăng red_team_alives/blue_team_alives nếu agent còn là key trong obss. PettingZoo 1.12.0 magent_env.step tính all_observes trước khi clear_dead và trước khi cập nhật self.agents. Agent vừa chết vẫn có thể xuất hiện trong dictionary observation trả về của bước đó. Vì vậy, wrapper có thể đếm số quân trước khi loại bỏ các ca chết ở bước cuối.

Probe tổng hợp với đúng _BattleWrapper.step tái hiện: observation vẫn có key cho ba agent đỏ, nhưng hai đã chết ở bước đó, đối diện hai agent xanh sống. Wrapper báo đỏ 3, xanh 2 và red_team_win=True, dù true counts là đỏ 1, xanh 2. Đây là phản ví dụ về logic metric, **không phải kết quả đo lại các episode W&B**.

Cần đối chiếu version magent_env.py thực tế trên cluster và log native get_num/get_alive sau khi xử lý dead tại terminal. Không suy ra toàn bộ 100 wins của Battle40 là giả chỉ từ phản ví dụ này. Export hiện tại không có per-episode native counts để hiệu chỉnh winner offline.

## 5. Vì sao Battle40 là ngoại lệ trong win rate?

### Yếu tố quan sát được: win criterion thô có thể che sự thay đổi lớn

Metric báo cáo của paper vẫn là Evaluation Win Rate. Các cột phụ sau chỉ phục vụ chẩn đoán và là số wrapper đã log, cần kiểm chứng true-alive như trên:

| Battle40, logged metric | Loss 0% | Loss 100% |
|---|---:|---:|
| Evaluation Win Rate | 1,00 | 1,00 |
| Mean red-team alives | 57,56 | 49,99 |
| Mean blue-team alives | 1,10 | 20,33 |
| Mean episode length (environment counter) | 83,31 | 194,55 |

Ngay cả khi giữ đúng định nghĩa thắng là red_alive > blue_alive, hai chiến thuật/kết cục rất khác nhau vẫn có thể cùng win rate 1,00. Dữ liệu tương thích với policy kết thúc chậm hơn và để lại nhiều đối thủ hơn ở loss100, nhưng vẫn hơn đối thủ về quân sống. Không có action traces để khẳng định nó cụ thể đã “camp”, “phòng thủ”, “ngừng tấn công” hay “thắng toàn bộ nhờ timeout”.

Đây là một yếu tố đã xác định ở cấp metric: win rate cao không đồng nghĩa không có tác động hành vi. Nó chưa giải thích đầy đủ vì sao checkpoint này giữ lợi thế quân số còn các checkpoint khác không giữ được.

### Yếu tố cấu trúc: bố trí ban đầu thực sự khác nhau

Chạy đúng generate_map với mock grid cho thấy:

| Map | N mỗi đội | Hình đội hình khởi tạo |
|---|---:|---|
| Battle25 | 20 | 4 cột × 5 hàng |
| Battle35 | 42 | 6 cột × 7 hàng |
| Battle40 | 64 | 8 cột × 8 hàng |

Không chỉ N thay đổi; hình đội hình, biên bản đồ và checkpoint đã train cũng khác. Chỉ Battle40 giữ đủ square ban đầu trong ba map này. Tương tác giữa hình đội hình, bên xuất phát cố định và pretrained opponent là giả thuyết có cơ sở source code, nhưng chưa có đối chứng để xác định cơ chế chiến thuật.

N=64 là lũy thừa của 2 không tự nó giải thích trực tiếp hiệu năng ở loss100, vì mọi remote link đều bị bỏ. Battle35 và Battle40 đều có cùng chu kỳ self-slot topk=7. Topology trong quá trình train có thể ảnh hưởng trọng số đã học, nhưng đó là câu hỏi khác cần đối chứng.

### Không tìm thấy bằng chứng global observation leak trong đường action

Mặc dù minimap_mode=True, get_obs của wrapper chỉ lấy hai kênh local team/enemy HP rồi flatten. Minimap được dùng trong get_state cho supervision; action controller lấy batch['obs'], không đưa global state trực tiếp vào agent. Vì vậy không có cơ sở từ đường code đã đọc để giải thích kết quả bằng việc actor vô tình nhìn toàn bộ bản đồ. Cần kiểm tra installation cluster nếu nghi ngờ có bản code khác.

## 6. Claim nào phù hợp với AdvPursuit?

**Claim có thể dùng với phạm vi rõ:**

> Under our fixed-opponent, single-seed evaluation, the trained AdversarialPursuit policies maintained high evaluation returns after all explicit inter-agent message delivery was disabled.

Số liệu loss 0% / 100% lần lượt là 87,38 / 83,00 (N=25), 73,81 / 75,55 (N=45), 65,87 / 68,56 (N=61). Claim nên công bố đủ sáu số này và protocol 100 episode, actual seed 0, một checkpoint/map, cùng pretrained prey.

**Chưa có bằng chứng cho claim tổng quát:** “Pursuit gần như không cần giao tiếp”, “communication không có ích trong training”, hoặc “mô hình không cần coordination”. Loss100 vẫn giữ self-link, own hidden state và message GRU ở self-slot. Policy đã train với communication. Agent còn có thể phối hợp ngầm qua vị trí/hành động quan sát được của đồng đội.

Benchmark này thưởng cho tagging chứ không đo trực tiếp tỷ lệ bắt hết unique prey. Config repo cộng +1 khi tag prey và trừ 0,2 cho mỗi hành động tag; mô tả benchmark cho phép tiếp tục tag prey mà không gây sát thương. Vì vậy, return cao không tự động chứng minh đã giải bài toán cooperative capture tổng quát. Local tagging hoặc phối hợp xuất hiện từ policy có thể đủ tốt trước pretrained prey này, nhưng cần traces để xác định hành vi.

## 7. Đối chứng ưu tiên

1. Sửa/kiểm chứng terminal alive counts bằng native MAgent counts. Log per episode old-winner và corrected-winner để biết bug có thực sự thay đổi kết luận không.
2. Đánh giá có kiểm soát cả hai bên xuất phát, tránh double reset. Chạy lại ít nhất loss 0%, 50%, 100% cùng protocol và native winner; không trộn baseline cũ với reset mới.
3. Sửa CLI seed, báo lỗi option không nhận diện, và thực sự chạy nhiều evaluation seed. Dùng nhiều checkpoint train độc lập trước khi khái quát về phương pháp.
4. Với Battle40, lưu action histogram, kill/attack/engagement theo thời gian, true counts, end reason và video ở loss0/loss100. Counterfactual đổi bên, đổi opponent và đổi bố trí map sẽ phân biệt giả thuyết hình đội hình/side bias khỏi policy độc lập communication.
5. Train baseline không communication để đo lợi ích của communication trong học. Thêm control cập nhật GRU với zero/shuffled input nếu muốn phân biệt giá trị nội dung message khỏi việc giữ bộ nhớ khi packet mất.

Không tự động sửa hoặc submit lại thí nghiệm trong audit này vì các sửa đổi làm thay đổi protocol và cần đối chiếu baseline đồng bộ.

## Nguồn

- Code local: src/config_overrides.py; src/main.py; src/modules/agents/ExpoComm_agent.py; src/controllers/ExpoComm_controller.py; src/runners/episode_runner.py; src/envs/battle_wrappers.py; src/envs/adv_pursuit_wrappers.py; env/battle_v3_view7.py; env/adversarial_pursuit_view8_v3.py.
- Dữ liệu: JSON gốc trong results/packet_loss100_20261110_155011_842018 và results/packet_loss_remaining_20261110_141557_124749; Battle25 baseline trong results/battle25_packet_loss.
- PettingZoo 1.12.0 source dùng để kiểm tra thứ tự reset/step: https://raw.githubusercontent.com/Farama-Foundation/PettingZoo/1.12.0/pettingzoo/magent/magent_env.py
- Mô tả benchmark Pursuit, tham khảo ngữ nghĩa tagging: https://magent2.farama.org/environments/adversarial_pursuit/ . Đây là tài liệu MAgent2 hiện tại; cấu hình/source của repo cũ mới là nguồn trực tiếp cho các tham số chạy.
- Bằng chứng probe: probe_results.json, tái tạo bằng `python scripts/audit_packet_loss_eval.py`.
