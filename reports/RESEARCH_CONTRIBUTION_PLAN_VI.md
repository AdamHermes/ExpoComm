# Kế hoạch contribution ExpoComm trong 4 ngày

Ngày khảo sát: 2026-11-12. Đây là đề xuất nghiên cứu dựa trên source local và tài liệu công khai; chưa implement phương pháp mới hoặc chạy checkpoint trên cluster trong lần khảo sát này.

## 1. Quyết định đề xuất

Tập trung vào **execution-time message recovery cho ExpoComm one-peer**, dùng weights hiện có. Câu hỏi chính: khi mất packet, nên đóng băng message memory, tiếp tục cập nhật từ thông tin riêng, hay cập nhật với message cũ còn đủ mới? Đánh giá cả mất gói độc lập và mất gói có tương quan theo thời gian.

Gói bắt buộc:

1. Evaluation protocol có seed thực, cân bằng bên, native alive counts và log từng episode.
2. Hold / Zero-update / Self-update / TTL-cache trên cùng checkpoint, topology, packet budget và channel traces.
3. Ablation cơ chế message-memory; IID versus burst; phân tích độ khác nhau giữa scenario/checkpoint.

Gói tùy chọn: một fine-tune message-dropout từ checkpoint, hoặc một adapter nhỏ học offline. Không đưa cả hai thành đường găng. Không train lại toàn bộ baseline để bắt đầu project.

Novelty phù hợp đồ án: thiết kế và kiểm chứng cách xử lý missing-message trên kiến trúc ExpoComm cụ thể, phân tách nguyên nhân và đánh giá có kiểm soát. Không khẳng định cache, dropout, Markov loss hay decentralized communication là phát minh mới.

Tên report gợi ý: **Beyond Packet-Loss Stress Tests: Execution-Time Message Recovery for ExpoComm**.

## 2. Những gì source hiện tại cho phép kết luận

### 2.1 Luồng thông tin

- `src/modules/agents/ExpoComm_agent.py`: `h` là hidden từ local observation; `msg` là message memory riêng. Q head dùng concat `[h, msg]`.
- `ExpoCommOAgent._communicate`: lấy message của sender ở bước trước, kết hợp với current local hidden qua `msg_processor`, rồi `msg_rnn` cập nhật memory.
- Khi remote packet mất, `th.where(keep_mask, candidate_msg, msg_ego)` giữ nguyên message memory. Local observation RNN vẫn cập nhật. Self-link không bị drop và vẫn có lần cập nhật message ở self-slot.
- `src/controllers/ExpoComm_controller.py`: sender luân phiên theo `t % topk_neighbors`, gồm offset 0 và các offset lũy thừa 2. Không có learned attention trong one-peer; không nên đưa proposal dựa trên sửa attention vào đường chính.
- Battle25 dùng IQL và auxiliary global-state reconstruction, không có QMIX mixer. Các config Pursuit tương ứng dùng QMIX/contrastive. Không gọi toàn bộ pipeline hiện tại là QMIX.
- State prediction head của reconstruction chỉ là auxiliary head. Dùng state thật ở evaluator để đo lỗi là hợp lệ; dùng lỗi với state thật để quyết định fallback lúc execution sẽ tạo privileged-information leak.
- Checkout local chỉ thấy các pretrained opponent `.pt`, không thấy learned `agent.th` checkpoints. Plan giả định nhóm truy cập được weights đã nói trên cluster.

### 2.2 Dữ liệu cũ định hướng chọn map, chưa thay thế evaluation mới

Theo `output/packet_loss_0_to100/ANALYSIS_VI.md`, các logged win rate tại loss 0/50/100%:

| Map | 0% | 50% | 100% |
|---|---:|---:|---:|
| Battle25 | 1.00 | 0.67 | 0.00 |
| Battle35 | 0.89 | 0.48 | 0.00 |
| Battle40 | 1.00 | 0.98 | 1.00 |

Pursuit return tương đối ổn định khi tắt remote delivery. Vì vậy chọn Battle25 và Battle35 làm hai scenario chính; dùng Battle40 và Pursuit45 để kiểm tra phạm vi hiệu quả, tránh chỉ chọn checkpoint có gain.

Theo audit có sẵn trong `reports/packet_loss/analysis/AUDIT_VI.md` và source hiện tại:

- `--eval-seed` chưa được parser hỗ trợ; tên run seed khác nhau không đảm bảo seed thực khác nhau.
- Wrapper reset simulator hai lần; điều này triệt tiêu việc luân phiên bên trong Battle.
- Đếm alive qua key trong observation có thể tính cả agent vừa chết ở terminal. Cần native counts sau xử lý dead.
- Các run lặp có actual seed 0 không phải nhiều training seed độc lập.

Không tự động kết luận Battle40 thắng do bug, cũng không coi audit tổng hợp là replay checkpoint. Giữ old metric để truy vết, đồng thời log corrected metric và đánh giá lại baseline cùng protocol mới.

## 3. Literature và ý nghĩa cho quyết định

| Công trình đã xác minh | Nội dung liên quan | Áp dụng trong 4 ngày |
|---|---|---|
| [ExpoComm — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3514dbacaebf0f38b25adfe59ed81a8a-Abstract-Conference.html) | Exponential topology, memory-based processing, message grounding. | Giữ backbone và weights; nghiên cứu fallback và information freshness. Contrastive grounding đã có trong baseline. |
| [Succinct and Robust Multi-Agent Communication With Temporal Message Control — NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/c82b013313066e0702d58dc70db033ca-Abstract.html) | Temporal locality, message buffers, expiration và regularization. | Cảm hứng cho cache theo sender và TTL. Chỉ chuyển buffering sang ExpoComm không phải reproduce đầy đủ TMC. |
| [Message-Dropout — AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4564) | Training với dropout cả block message để tăng khả năng chịu lỗi communication. | Fine-tune dropout là baseline training phù hợp. Hold-GRU trong repo khác phép dropout/compensation gốc; ghi rõ là adaptation. |
| [Multi-Agent Reinforcement Learning with Communication-Constrained Priors — NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/e26502ce357ce3015e8778f0e85d4b39-Paper-Conference.pdf) | Mô hình channel constraints và learning priors, phân biệt tác động message dưới điều kiện lỗi. | Cơ sở đánh giá channel có tương quan. Không có novelty nếu chỉ thay IID bằng Markov; full mutual-information framework quá lớn. |
| [DACOM — AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26389) | Học cách điều chỉnh thời gian chờ nhận communication. | Động lực đo latency/freshness. TimeNet không phải module dùng ngay với checkpoint ExpoComm. |
| [ADMAC — AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29708) | Ước lượng độ tin cậy của message và điều chỉnh tác động lên decision. | Cảm hứng giới hạn tác động stale messages. Không thể gắn learned reliability estimator chưa train rồi kỳ vọng chạy được. |
| [Learning Multi-Agent Communication with Contrastive Learning — ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/de2ad3ed44ee4e675b3be42aa0b615d0-Abstract-Conference.html) | Contrastive learning tạo message representations phục vụ coordination. | Dùng làm related work/probing; repo đã có contrastive agent, không ghi là contribution mới. |
| [Learning Attentional Communication for Multi-Agent Cooperation — NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/6a8018b3a00b69c008601b8becae392b-Abstract.html) | Học khi nào cần giao tiếp và tích hợp thông tin. | Hướng event-triggered/importance scheduling dài hạn; thay architecture sẽ cần train và tuning. |
| [Multiagent Planning under Uncertainty with Stochastic Communication Delays — ICAPS 2008](https://people.csail.mit.edu/fao/docs/Spaan08ICAPS.pdf) | Dec-POMDP với communication delay ngẫu nhiên và xác suất truyền thành công. | Nền tảng formalize communication không lý tưởng; solver planning không thay trực tiếp neural controller. Không gán nhầm venue cho các bài MAPF. |
| [SOPS — ICML 2026](https://proceedings.mlr.press/v306/deng26n.html) | Learned sparse link allocation trên exponential backbone. | Related work rất gần về topology; repo đã có thư mục SOPS. Không chọn learned topology làm contribution chính trong 4 ngày. |

Đây là khảo sát có chọn lọc để ra quyết định thực hiện; chưa đủ để tuyên bố phương pháp đầu tiên hoặc novelty cấp hội nghị. Những rule fallback dưới đây là đề xuất của plan, không phải phương pháp được các bài trên bảo đảm cải thiện.

## 4. Contribution chính: tách missing information khỏi missing memory update

Ký hiệu `F(h, r; m)` là `msg_rnn(msg_processor([h, r]), m)` trong one-peer. `h_i^t` là local hidden hiện tại, `m_i^{t-1}` là memory riêng, sender theo lịch là `j`, và message đang truyền là `m_j^{t-1}`.

Khi nhận thành công, mọi phương pháp dùng nguyên bản:

`m_i^t = F(h_i^t, m_j^{t-1}; m_i^{t-1})`.

Khi remote packet mất:

| Method | Cập nhật | Vai trò |
|---|---|---|
| Hold | `m_i^t = m_i^{t-1}` | Packet-loss baseline hiện tại. |
| Zero-update | `m_i^t = F(h_i^t, 0; m_i^{t-1})` | Giữ nhịp cập nhật GRU, không dùng remote content. |
| Self-update | `m_i^t = F(h_i^t, m_i^{t-1}; m_i^{t-1})` | Dùng operation self-slot mà checkpoint từng gặp; không cần parameter mới. |
| TTL-cache + Self | Nếu cache đúng sender còn hạn: `F(h_i^t, c_ij; m_i^{t-1})`; nếu chưa có/hết hạn: Self-update. | Kiểm tra giá trị message cũ sau khi đã control việc cập nhật bộ nhớ. |

Tất cả giữ nguyên self-slot và đường nhận thành công. Quyết định fallback chỉ cần local hidden, local memory, sender theo lịch, cache nhận hợp lệ và timestamp tại receiver. Không cần global state hoặc truy cập live hidden của sender khi packet mất.

### 4.1 Câu hỏi và cách diễn giải

- Self-update hơn Hold: kết quả tương thích với lợi ích duy trì local message-memory dynamics khi mất gói; chưa chứng minh đã phục hồi thông tin đồng đội.
- TTL-cache hơn Self-update: cung cấp bằng chứng stale remote content có ích trong cách dùng cụ thể, ngoài việc cập nhật memory.
- Zero-update tốt tương đương Self: local hidden và nhịp cập nhật có thể giải thích phần lớn gain trong checkpoint này.
- Hold tốt nhất: fallback chưa train có thể gây distribution shift; đây là kết quả phủ định cần báo cáo.
- Cache có ích với IID nhưng kém dưới burst: chỉ kết luận khi có số liệu, có thể phân tích expiration và thời gian thiếu remote updates.

Phân tích paired action disagreement hoặc Q differences trên cùng local history có ích để chẩn đoán; không thay thế rollout thực tế. Không gán causal effect của raw message content một cách tuyệt đối khi memory trajectories cũng thay đổi.

### 4.2 Cache cần làm đúng

1. Cache theo `(receiver, sender)` hoặc `(receiver, exponential-offset slot)`, không dùng một cache chung cho mọi sender luân phiên.
2. Chỉ ghi cache khi packet đến thành công; lưu payload `m_j^{t-1}` thật đã nhận, không ghi `candidate_msg` của receiver vào cache sender.
3. Reset cache, valid bit, timestamp và channel state ở đầu episode; tuổi tăng theo environment timestep.
4. Định nghĩa tuổi là thời gian từ lúc receiver nhận payload gần nhất. Payload mới vốn đã lấy từ timestep trước; đây không phải tuổi của toàn bộ thông tin global trong latent.
5. TTL chính đặt `K = topk_neighbors` bước; sensitivity trên validation với `2K` và cache không hết hạn. Cùng sender thường chỉ được schedule lại sau K bước, nên tránh TTL nhỏ hơn K khiến cache hầu như không dùng được. Định nghĩa rõ điều kiện `age <= TTL`.
6. Cache hết hạn hoặc chưa có thì Self-update. Không tự đi lấy fresh tensor của sender từ batch khi packet mất.
7. Snapshot messages của toàn bộ agent trước communication; không cập nhật lần lượt để agent phía sau đọc message mới cùng bước.
8. Cache chỉ lưu exponential candidates, bộ nhớ thêm `O(N K d)`, không cần full `O(N^2 d)` cache. Message bandwidth giữ nguyên. Đo thêm memory thực tế và inference latency.
9. Native alive mask dùng cho metric; nếu đưa alive status toàn cục vào scheduler/fallback thì phải chứng minh thông tin đó có sẵn tại runtime. Các comparison chính giữ cùng quy ước dead-agent slots của baseline.

### 4.3 Sanity checks nhỏ nhưng cần thiết

- p=0: cùng Q/action/rollout với nguyên bản khi RNG/environment giống nhau.
- p=1: không đọc remote payload mới; cache không từng có remote success; cache method trùng Self-update nếu cùng initial state.
- Sender identity chính xác trong ít nhất hai chu kỳ; cache không lọt qua episode reset.
- RNG channel riêng không tiêu thụ RNG của opponent/environment.
- Đường cuối episode của runner gọi action thêm để ghi batch: không tính lần gọi đó vào packets/latency thực thi.
- Log packet attempts, deliveries và drops trên remote opportunities; self-links không nằm trong mẫu số.

## 5. Contribution đánh giá: loss pattern và freshness

### 5.1 Channel IID

Giữ nguyên quy tắc whole-message Bernoulli loss và self-link exemption để so được với baseline. Test p chính: 0, 0.3, 0.5, 0.7, 1.0. Nếu thiếu compute: 0, 0.5, 1.0 trước, sau đó thêm hai điểm.

### 5.2 Channel burst có cùng stationary loss

Dùng channel hai trạng thái theo receiver: Good nhận thành công; Bad drop mọi incoming remote packet. State chuyển mỗi environment timestep kể cả self-slot; bắt đầu episode từ stationary distribution. Đây là receiver outage, khác mô hình Markov per-link và cần ghi đúng tên.

Đặt `a=P(G→B)`, `b=P(B→G)`:

`p_stationary = a/(a+b)`; `E[bad-run length]=1/b`.

Với desired mean bad-state duration `L`: `b=1/L`, `a=p/((1-p)L)`. Kiểm tra `a,b` trong [0,1]. p=0 và p=1 là special cases.

Ví dụ: p=0.5, L=4 → a=b=0.25; L=8 → a=b=0.125. Đánh giá chính p=0.5, L∈{4,8}; so IID cùng p. Đừng gọi L là độ dài chuỗi lost packets thực tế vì self-slots xen vào và không truyền packet. Log cả bad-state duration lẫn observed lost-packet run length và realized loss.

Ý nghĩa: [RFC 6534](https://www.rfc-editor.org/rfc/rfc6534.html) lưu ý tác động ứng dụng phụ thuộc cách packet losses tập trung thành loss episodes, không chỉ mean loss. NeurIPS 2025 về Communication-Constrained Priors đã nghiên cứu Markov constraints; đóng góp của nhóm là benchmark/ablation cụ thể trên ExpoComm và fallback, không phải phát minh mô hình burst.

### 5.3 Diagnostic blackout và topology

Nếu có thời gian: tắt remote communication đồng loạt trong một cửa sổ cố định, ví dụ 10 hoặc 20 bước sau warm-up. Vẽ hiệu năng/memory-update/action theo thời gian trước-trong-sau blackout. Không trộn blackout có warm-up với p=1 từ đầu episode.

Có thể làm simulator information-reachability trên lịch exponential để đo thời gian provenance đi qua các link dưới loss; chỉ là tiềm năng truyền thông trên graph, không chứng minh latent neural message chứa hoặc giữ nguyên thông tin. Không cần làm để hoàn thành gói chính.

## 6. Những hướng execution khác và mức ưu tiên

| Ý tưởng | Khả thi | Quyết định |
|---|---|---|
| Hold/Self/Cache ablation | Cao, không train, không tăng network attempts | Chính. |
| IID versus burst, blackout | Cao, cần channel state đúng | Chính cho burst; blackout tùy chọn. |
| Fixed duplication 2 copies | Dễ implement | Chỉ baseline đánh đổi communication-cost; không tự thân novelty. |
| Event-triggered duplication theo local message novelty | Trung bình | Stretch nếu hết ngày 2 đã có kết quả chính. |
| Đảo thứ tự offset/chọn offset khác | Dễ code, khó diễn giải | Chỉ appendix: latent distribution shift, không có bảo đảm tốt hơn. |
| Đổi sang static attention/topology mới | Cần weights tương thích hoặc train lại | Không đi đường chính. |
| Learned reliability gate/GNN router | Nhiều training và tuning | Future work. |

Với duplication, nếu độc lập giữa các copies thì loss hiệu dụng `p^r`; không đúng nếu chúng nằm chung outage hoặc bị drop đồng thời. Tính mọi copies vào attempted packets/bytes, không cho GRU cập nhật nhiều lần trên cùng logical message, và công bố timing model. Không so performance 2 copies với 1 copy rồi gọi là vượt trội cùng budget. Cần fixed/random budget-matched controls nếu claim adaptive allocation tốt hơn.

Event-triggered sender rule có thể dùng độ đổi message riêng từ hai bước trước; nếu dùng receiver cache age thì cần feedback/request và tính overhead. Q margin chưa được calibration không phải xác suất chắc chắn. Không dùng attention entropy hoặc oracle task state làm tín hiệu miễn phí cho one-peer.

## 7. Training tùy chọn A: một fine-tune với message loss

Đây là lựa chọn ưu tiên nếu nhóm cần có training extension và có compute riêng. Là adaptation baseline từ Message-Dropout, không gọi dropout là ý tưởng mới.

### 7.1 Thiết kế cố định trước chạy

- Chỉ Battle25 ở lượt đầu, vì checkpoint có sensitivity trong log cũ và số agent nhỏ.
- Khởi tạo từ cùng checkpoint, không đổi architecture.
- Hai continuation để diễn giải hợp lý: clean fine-tune và dropout fine-tune với **cùng số environment steps/update budget**, nếu compute cho phép.
- Dropout condition cố định p_train=0.3; test trên cùng grid loss, gồm burst chưa train. Giữ fallback Hold để tách tác dụng training khỏi recovery.
- Khởi điểm lr=0.25×lr cũ; đây là heuristic chốt trước, không bảo đảm tối ưu. Giới hạn 0.5M–1M additional environment steps sau khi đo throughput; không cam kết runtime chỉ dựa trên tỷ lệ steps.
- Save nhiều checkpoint theo lịch định trước. Chọn checkpoint qua validation metric tổng hợp clean/loss, không chọn best test điểm riêng.

### 7.2 Rủi ro resume nhìn thấy trong source

- `src/run.py` đặt `runner.t_env=timestep_to_load`; `t_max` là absolute cap, phải lớn hơn load step, ví dụ load 5,009,240 cộng 500,000.
- Replay buffer được tạo mới và cần warm-up; full episode replay training có chi phí đáng kể.
- `load_models` tải optimizer state; đặt lr trước load có thể bị optimizer state ghi đè. Nếu đổi lr phải chỉnh param_groups sau load hoặc chủ động reset optimizer, ghi rõ lựa chọn.
- RunningMeanStd của rewards không được `save_models`/`load_models` hiện tại lưu/khôi phục. Resume khác original training trajectory; cân nhắc calibration cố định trước update và áp dụng giống nhau cho cả continuation controls.
- Target mạng được copy từ online khi load; các training counters không được resume toàn bộ.
- Packet masks lấy lại khi forward replay, không được lưu như part của episode batch. Nếu thay mask semantics phải kiểm tra online/target alignment và sequence reset, không giả định dropout ở rollout là toàn bộ training protocol.
- Khi báo cáo gọi đây là warm-start fine-tuning trong protocol đã định nghĩa, không gọi là exact resume.

Nếu chỉ đủ một dropout continuation, vẫn có thể so với checkpoint gốc nhưng không tách được robustness gain do thêm training khỏi gain do dropout. Ghi rõ limitation; không che bằng nhiều evaluation seeds.

Nếu không có reward normalizer gốc và continuation không ổn trong smoke run, giữ đây là unfinished extension; đừng dành cả 4 ngày sửa resume.

## 8. Training tùy chọn B: adapter bù message học offline

Chọn B thay A nếu nhóm muốn tránh environment RL training dài, nhưng chỉ khi có người triển khai sequence training tự tin. Không gọi là training-free.

1. Thu 50–100 trajectories từ frozen teacher trên train episode seeds riêng; lưu local obs/actions, sender schedule, received payload targets và clean-teacher Q.
2. Replay corruption tuần tự để tạo student history riêng. Không đưa clean teacher memory/history vào adapter input nếu student không có lúc execution.
3. Freeze baseline; MLP nhỏ dự đoán remote message hoặc candidate memory khi drop từ `h_local`, `m_local`, cache đúng sender, valid bit, age và offset. Không lấy local obs của agent khác làm runtime input.
4. Khởi đầu residual/correction bằng 0 hoặc có bypass về Self-update; giữ đường packet-success nguyên bản.
5. Supervision có thể dùng MSE message và Q/action consistency với teacher trên cùng recorded observation sequence. Q logits không phải calibrated probabilities; nếu dùng KL phải ghi temperature và kiểm chứng. Không ép MSE clean target như thể luôn suy ra được remote information từ local history.
6. Split theo episode, không random-split timesteps. Teacher dùng full communication để sinh label trong training là hợp lệ; teacher không được chạy giúp student lúc evaluation.
7. Đánh giá **closed-loop** bằng student execution. Loss offline thấp hoặc action agreement trên teacher trajectories không chứng minh return tốt; có distribution shift khi student điều khiển môi trường.

Giới hạn fixed budget: thu data + train tối đa nửa ngày đến một ngày theo throughput đo được; một architecture nhỏ, một seed; nếu thất bại chuyển thành pilot/future work. Không triển khai adapter lẫn fine-tune RL đồng thời nếu chung nhân lực/compute.

## 9. Evaluation protocol đề xuất

### 9.1 Sửa những gì trước khi đo

- Parser hỗ trợ seed đúng và fail khi còn unknown flags; log seed effective, không chỉ tên run.
- Tách RNG environment, opponent nếu có randomness, channel và training. Mọi method được chạy lại env độc lập với cùng initial seed/side.
- Chạy balanced starting sides, giữ side như một field trong per-episode log. Sửa double reset rồi chạy lại cả baseline và methods. Nếu env không đổi bên có kiểm soát được, không claim side-balanced.
- Native alive counts sau dead removal; log old/native winner song song, kill counts, normalized survivor difference và timeout/end reason.
- Freeze opponent, checkpoint và code revision; hash checkpoint, ghi dependency versions. Giữ protocol cũ trong dữ liệu cũ.
- Lưu đủ per-episode log trước khi aggregate; tắt recording trong sweep, chỉ quay một số episode định sẵn.

### 9.2 Ma trận tối thiểu và mở rộng

**Pilot/validation:** Battle25, 4 methods, p∈{0,0.5,1}, 10–20 episodes/condition với seed riêng. Chọn TTL và đóng code sau pilot; không lấy pilot CI làm kết quả chính.

**Main:** 2 maps × 4 methods × 5 loss levels × 3 evaluation seeds × 50 episodes = **6,000 episodes**. Mỗi seed 50 episode, chia 25/25 bên nếu kiểm soát được. Có thể dùng shared p=0 result sau khi đã xác nhận methods giống hệt đường no-loss; ghi rõ reuse.

**Burst:** 2 maps × 3 methods (Hold, Self, Cache) × 2 bad-state durations × 3 eval seeds × 50 episodes = **1,800 episodes** tại p=0.5; dùng IID p=0.5 ở main làm đối chứng.

**Scope check:** Battle40 và Pursuit45 với Hold/selected recovery tại p=0,0.5,1. Nếu không đủ thời gian, 30–50 episodes/condition và ghi exploratory.

Không cần chạy đủ sáu map ngay. Nếu throughput thấp: main còn p=0,0.5,1, ưu tiên 100 episodes/điều kiện chính hơn chạy nhiều biến thể với rất ít episode. Đừng chọn lại method dựa trên test map rồi báo như held-out generalization.

### 9.3 Thống kê và công bằng

- Eval seeds khác nhau đo deployment variability; không thay thế training seeds. Với một checkpoint/map, mọi kết luận là conditional on those checkpoints.
- Reuse channel traces indexed theo receiver/time cho tất cả methods; đối với topology/redundancy experiments cần định nghĩa trace theo edge/attempt phù hợp. Khi policies phân kỳ, common trace không làm physical trajectories giống nhau.
- Ở p=0.5 một sample 50 games có uncertainty lớn; gain vài điểm win rate chưa đủ kết luận. Tăng episode cho contrasts chính nếu CI quá rộng.
- Win rate báo Wilson CI hoặc bootstrap theo episode; paired difference bootstrap giữ cặp seed/side/episode. Nếu episodes không độc lập do reset/RNG state, bootstrap theo block phù hợp hoặc nêu giới hạn.
- Bảo toàn tính paired giữa methods, phân tầng theo side và seed; hiển thị cả seed-level results. Chỉ 3 eval seeds thì CI giữa seed rất yếu, không trình bày như 3 training runs.
- Không xem agents/timesteps trong cùng episode là các mẫu độc lập cho CI reward/win.
- Không đánh đồng 100% loss sau warm-up với 100% từ đầu; self links không tính vào delivered remote packets.

### 9.4 Metrics và figures

Primary Battle: native-alive-based win rate với CI. Secondary: mean return, survivor difference `(red-blue)/N`, kill count, timeout rate, environment steps/episode. Pursuit giữ evaluation return theo baseline; không đổi return thành capture success khi không đo unique prey capture.

Communication: attempted packets, delivered packets, realized loss, bytes/payload dimension, time since last successful remote receive, cache age/expiry/hit fraction, message update fraction. Self-update không tạo remote information mới nên receiver freshness không reset chỉ vì local fallback đã chạy.

Mechanism: Hold/Self/Cache action disagreement trên common histories; message-update fraction; lỗi global-state reconstruction chỉ với agent có head được train cho mục tiêu đó. Contrastive agent trả normalized latent, không phải global-state prediction.

Figures đủ cho report:

1. Main robustness curves: hai panel Battle25/35, bốn methods, loss axis và CI.
2. IID versus burst bars hoặc lines tại matched mean loss.
3. Ablation Hold → Zero/Self → Cache; đi cùng update fraction/cache-hit statistics.
4. Survivor difference và episode length ở Battle40, tránh win-rate ceiling che mất tác động.
5. Nếu có training: original/clean continuation/dropout continuation với cùng additional steps; loss ở test cố định.

Không tạo CI từ bảng means W&B cũ. Không vẽ anticipated curves như measured results.

## 10. Ngân sách và lịch thực hiện

Trước launch chính: đo median và tail episode seconds từ pilot có rendering tắt. Runtime thô = episode count × measured seconds/episode / số workers độc lập hiệu dụng, cộng 20–30% dự phòng. Benchmark lại khi tăng concurrency vì CPU simulator hoặc RAM có thể bottleneck.

Ví dụ 6,000 episodes trên 1 worker: 2 giây/episode → 3.3 giờ; 10 giây → 16.7 giờ; 20 giây → 33.3 giờ, chưa overhead. Đây là phép tính kế hoạch, không phải số đo trên cluster của nhóm.

| Thời gian | Việc cần xong | Deliverable/điểm dừng |
|---|---|---|
| Ngày 1, 0–4h | Protocol/seed/alive/side; locate weights; đo throughput | Log một episode đầy đủ, smoke check no-loss. |
| Ngày 1, 4–10h | Hold/Zero/Self/cache; deterministic checks; pilot validation | Một figure pilot và config đã chốt. Nếu native metric chưa đáng tin, ưu tiên sửa protocol, thu hẹp methods. |
| Cuối ngày 1 | Đóng execution design; launch main jobs | Manifest jobs, resume theo job, per-episode JSONL. Training optional chỉ launch nếu smoke pass và có compute riêng. |
| Ngày 2 | Main sweep + burst; viết method/related work | Dữ liệu hai map; không thêm architecture lớn. Nếu không gain, giữ nghiên cứu ablation/failure analysis. |
| Ngày 3 | CI, diagnostics Battle40/Pursuit, videos; training result nếu có | Figures và captions; freeze experiments chính trước cuối ngày. |
| Ngày 4 | Viết/soát report, reproducibility, bảng trạng thái, consistency checks | Report hoàn chỉnh; dành thời gian buffer, không phụ thuộc job 48h mới. |

Nếu nhóm ba người: người 1 protocol+recovery, người 2 jobs+analysis, người 3 literature+report và training optional sau smoke. Nếu chỉ một người implement: bỏ adapter, bỏ adaptive duplication, giữ main+burst trước.

**Stop rules:** không tunetest nhiều TTL; sau ngày 2 không mở phương pháp mới; training không ổn không được làm chậm report; không có improvement vẫn báo negative results và lý do đã có kiểm chứng. Không hứa với TA trước rằng recovery chắc chắn hơn baseline.

## 11. Implementation map cho lần thực hiện tiếp theo

| File/module | Thay đổi dự kiến |
|---|---|
| `src/modules/agents/ExpoComm_agent.py` | Missing-packet policy và cache hooks cho OAgent; không đổi state_dict của policy chính. |
| `src/controllers/ExpoComm_controller.py` | Reset cache/channel state, sender/offset/time hooks; preserve synchronous semantics. |
| Module channel riêng đề xuất | IID/Markov state và RNG riêng, export trace, diagnostic counters. |
| `src/config_overrides.py`, `src/main.py` | Explicit seed/method/channel arguments, unknown-flag checking, effective config. |
| `src/envs/battle_wrappers.py` | Controlled reset/sides, true-alive metric và diagnostic old metric. |
| `src/runners/episode_runner.py` | Per-episode JSONL; exclude terminal-only action call from communication counters. |
| Script sweep mới | Job manifest, independent seeds, method grid, failure detection/resume. |
| Script plot mới | Episode-level aggregation, paired/stratified CI, provenance và captions. |

Đây là bản đồ thay đổi, chưa phải các flags có sẵn để copy chạy. Không cung cấp lệnh với unsupported flags như thể repo đã implement.

## 12. Cách viết contribution trung thực

Ba claim ở trạng thái **dự kiến**, chỉ chuyển sang past tense khi đã thực hiện:

1. We develop and evaluate checkpoint-compatible execution-time message recovery rules for ExpoComm under unreliable communication.
2. We investigate the distinction between missing remote information and skipped message-memory updates through controlled ablations.
3. We evaluate independent and temporally correlated communication failures using a corrected, reproducible protocol and episode-level diagnostics.

Nếu đã có số liệu tốt: thêm gain, CI và overhead thật. Nếu chỉ có pilot: ghi rõ map/checkpoint/episodes, không suy rộng. Nếu chưa chạy: trình bày thuật toán và hypothesis trong Proposed Extensions/Future Work, không đặt trong Results hoặc tô thành đường cong thực nghiệm.

Không ghi: first robust ExpoComm; novel caching; improved general robustness sau một checkpoint; communication unnecessary chỉ từ Pursuit; hay training-free nếu có học adapter/calibration tối ưu trên test.

Structure report: reproduction và setup → audit/evaluation protocol → research questions → recovery rules → results+ablation → limitations/unfinished training extension. Phân biệt borrowed idea, adaptation, implementation, empirical finding và proposed future work.

**Mức đủ mạnh trong 4 ngày:** hai map chính với main+burst, recovery ablation có controls, log đáng tin và một phân tích scenario không hưởng lợi. Đây là contribution thực nghiệm và phương pháp nhỏ có thể bảo vệ được cho đồ án, dù không đủ cơ sở tự động tuyên bố novelty cấp hội nghị.
