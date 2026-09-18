# Thuật toán làm ra dựa trên bài báo sau:
Chen, Xiangchun, et al. "Dynamic task offloading in edge computing based on dependency-aware reinforcement learning." IEEE Transactions on Cloud Computing 12.2 (2024): 594-608.

# Hướng dẫn ở dưới đây là dành cho hệ điều hành Linux + chạy trên GPU cục bộ
# ADPRL — Triển khai PyTorch

Triển khai thuật toán **ADPRL (Asynchronous Deep Progressive Reinforcement Learning)**
trong bài báo:

> Chen, Cao, Sahni, Jiang, Liang — *"Dynamic Task Offloading in Edge Computing based on
> Dependency-aware Reinforcement Learning"*, IEEE Transactions on Cloud Computing, 2024.

Bài báo giải quyết bài toán **ODTO (Online Dependent Task Offloading)**: đồng thời
quyết định (1) offload subtask nào đến node biên nào, và (2) cấp phát băng thông cho
luồng dữ liệu tương ứng, để tối ưu thời gian hoàn thành tác vụ (ACT) và năng lượng
tiêu thụ (EC), có xét đến **phụ thuộc giữa các subtask (DAG)**.

---

## 1. Cấu trúc code

```
adprl_project/
├── dag_generator.py   # Sinh tác vụ dạng DAG (Section 6.1 "Synthetic Dataset")
├── edge_env.py         # Môi trường mô phỏng CEC: node, link, Eq.(1)-(11) trong bài báo
├── networks.py         # Actor / Critic (PyTorch)
├── ddpg_agent.py        # Agent ADPRL = DDPG + replay buffer + async rollout (Algorithm 1)
├── baselines.py         # Random / LE / Greedy / DQN+FCFS (4 baseline trong bài báo)
├── train.py              # Script huấn luyện ADPRL (mục tiêu LO hoặc EE)
├── evaluate.py            # So sánh ADPRL vs baseline, quét tham số, vẽ hình như Fig.6/8/9/10
├── requirements.txt
└── README.md
```

### Ánh xạ thành phần bài báo → code

| Bài báo | File / hàm |
|---|---|
| Eq. (1) Task Computation Time | `edge_env.CECEnv.step` (phần `comp_time`) |
| Eq. (3)-(5) Flow communication time, finish time | `edge_env.CECEnv.step` (phần `comm_time`, `finish`) |
| Eq. (6)-(10) Energy consumption | `edge_env.CECEnv.step` (phần `ec_comp`, `ec_tx`) |
| Eq. (11)/(23) QoS / reward | `edge_env.CECEnv._step_reward`, `_finalize_reward` |
| State space S = (G, E, A) (Sec 5.2) | `edge_env.CECEnv._observe` |
| Action space (Sec 5.3) | `networks.Actor`, `edge_env.CECEnv.step(action)` |
| Reward function (Sec 5.4) | `edge_env.CECEnv._step_reward` |
| DDPG / Algorithm 1 | `ddpg_agent.ADPRLAgent` |
| "Asynchronous" nhiều worker (server=1, worker=4) | `train.py` (nhiều `CECEnv` song song, 1 replay buffer chung) |
| 4 baseline (Sec 6.1) | `baselines.py` |
| λt, λe cho LO / EE (Sec 6.1) | tham số `--objective LO/EE` trong `train.py` |

---

## 2. Cài đặt môi trường

Bạn nói đã cài CUDA và sẵn sàng chạy GPU — chỉ cần cài các gói Python sau (khuyến
nghị dùng virtualenv/conda riêng):

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Cài PyTorch có CUDA — chọn đúng lệnh theo phiên bản CUDA của bạn tại
# https://pytorch.org/get-started/locally/
# Ví dụ với CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Các gói còn lại
pip install -r requirements.txt
```

Kiểm tra GPU:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Nếu in ra `True` và tên GPU thì đã sẵn sàng.

---

## 3. Huấn luyện ADPRL

Bài báo huấn luyện **2 model**: một cho mục tiêu **LO** (Latency-Optimized,
λt=1.0, λe=0.0) và một cho **EE** (Energy-Efficient, λt=0.5, λe=0.5) — xem Sec 6.1.

```bash
mkdir -p runs

# Model tối ưu độ trễ (dùng để so sánh ACT, giống các hình 6a, 9)
python3 train.py --objective LO \
    --episodes 3000 \
    --tasks-per-episode 20 \
    --edge-nodes 25 \
    --workers 4 \
    --batch-size 64 \
    --buffer-size 10000 \
    --actor-lr 0.001 \
    --critic-lr 0.002 \
    --device cuda \
    --out runs/lo_model.pt

# Model tối ưu năng lượng (dùng để so sánh EC, giống các hình 6b, 10)
python3 train.py --objective EE \
    --episodes 3000 \
    --tasks-per-episode 20 \
    --edge-nodes 25 \
    --workers 4 \
    --batch-size 64 \
    --buffer-size 10000 \
    --actor-lr 0.001 \
    --critic-lr 0.002 \
    --device cuda \
    --out runs/ee_model.pt
```

Các siêu tham số mặc định (`--batch-size 64`, `--buffer-size 10000`,
`--actor-lr 0.001`, `--critic-lr 0.002`) **lấy đúng theo Sec 6.1 của bài báo**.
Tham số `--gamma` (reward decay) bài báo ghi là `0.001`; đây là giá trị khá bất
thường cho một discount factor RL thông thường — mình để mặc định đúng theo bài
báo nhưng bạn có thể thử `--gamma 0.99` nếu thấy huấn luyện học chậm/không ổn định
với chân trời (horizon) dài.

Trong lúc chạy, log in ra mỗi `--log-every` episode:

```
ep    20/3000  reward= -12.531  ACT=  842.11  EC= 145.32  OR= 0.62  elapsed=   4.3s
```

- `reward`: phần thưởng trung bình / episode (càng gần 0 càng tốt, vì reward luôn âm)
- `ACT`: Average Completion Time của episode đó (ms, đơn vị mô phỏng)
- `EC`: tổng năng lượng tiêu thụ
- `OR`: Offloading Ratio (tỉ lệ subtask được offload ra ngoài node nguồn)

Sau khi chạy xong, bạn có:

```
runs/lo_model.pt              # trọng số actor + critic
runs/lo_model_history.json    # lịch sử reward/ACT/EC/OR theo episode (để vẽ đường học)
```

**Thời gian huấn luyện tham khảo** (RTX 3060, 25 node, 20 task/episode,
12 subtask/task trung bình): ~3000 episode ≈ 10–20 phút trên GPU. Tăng
`--episodes` lên 10000+ để hội tụ tốt hơn, giống "training converges at events
for around 10.000" được nêu trong bài báo (Sec 6.1).

---

## 4. Đánh giá / So sánh với baseline (tái tạo Fig. 6, 8, 9, 10)

Sau khi có `runs/lo_model.pt` và `runs/ee_model.pt`:

```bash
python3 evaluate.py \
    --lo-model runs/lo_model.pt \
    --ee-model runs/ee_model.pt \
    --out-dir results/
```

Muốn chạy nhanh để kiểm tra pipeline hoạt động đúng trước khi chạy full:

```bash
python3 evaluate.py --lo-model runs/lo_model.pt --ee-model runs/ee_model.pt \
    --out-dir results_quick/ --quick
```

Script sẽ:

1. Quét **số lượng task** [10, 20, 30, 40] → `ACT_vs_num_tasks.png`, `EC_vs_num_tasks.png`
   (tương ứng Fig. 9a/b, 10a/b)
2. Quét **số lượng subtask/task** [25, 50, 75, 100] → tương ứng Fig. 9c/d, 10c/d
3. Quét **băng thông trung bình** [2, 4, 6, 8] Mbps → tương ứng Fig. 9e/f, 10e/f
4. Quét **tốc độ xử lý trung bình** [10, 20, 30, 40] Mcps → tương ứng Fig. 9g/h, 10g/h
5. Quét **số lượng node biên** [25, 50, 75, 100] → tương ứng Fig. 9i/j, 10i/j

Mỗi lần quét xuất ra 1 file `.csv` (số liệu thô) và 2 file `.png` (ACT, EC) trong
`results/`, cho cả 6 thuật toán: `Random, LE, Greedy, DQN+FCFS, ADPRL (LO), ADPRL (EE)`.

### Lưu ý quan trọng về việc quét "số lượng node biên"

Kiến trúc Actor có **một đầu ra softmax kích thước = số node biên lúc train**
(giống hầu hết cách hiện thực DDPG cho action rời rạc). Vì vậy 1 model ADPRL đã
train với `--edge-nodes 25` **chỉ đánh giá được trên môi trường có đúng 25 node**.//
Khi `evaluate.py` quét `num_nodes` sang 50/75/100, các điểm dữ liệu của ADPRL tại
những giá trị khác 25 sẽ bị bỏ qua (NaN) — baseline vẫn chạy bình thường.

Để có đường cong đầy đủ cho ADPRL trên trục "số node biên" giống Fig. 9i/9j, 10i/10j,
bạn cần train thêm các model riêng cho từng số node:

```bash
for N in 25 50 75 100; do
  python3 train.py --objective LO --edge-nodes $N --episodes 3000 \
      --out runs/lo_model_${N}nodes.pt
done
```
rồi sửa `evaluate.py` (hàm `sweep_and_plot` cho `num_nodes`) để nạp đúng model theo
từng giá trị N — phần này để trống có chủ đích vì tùy bạn muốn cố định cấu trúc
action space theo cách nào (ví dụ: đặt `--edge-nodes` = số node lớn nhất có thể gặp,
rồi mask các node không tồn tại trong môi trường nhỏ hơn).

---

## 5. Những điểm được đơn giản hóa / khác với bài báo (đọc kỹ trước khi so kết quả)

Bài báo không công bố source code, và nhiều chi tiết triển khai (bộ sinh topology
mạng chính xác, cách bộ sinh DAG ngẫu nhiên hoạt động, các hệ số công suất chính xác,
kiến trúc mạng nơ-ron chi tiết của DQN baseline, v.v.) không được mô tả đầy đủ trong
bài báo. Bản triển khai này **bám sát các công thức toán học (Eq. 1–30), Algorithm 1,
và các mô tả định tính** của bài báo, nhưng có một số đơn giản hóa cần lưu ý:

1. **"Asynchronous" worker**: bài báo dùng 1 server + 4 worker thực sự chạy song song
   (multi-process/multi-machine). Ở đây mình mô phỏng hiệu ứng đó bằng cách luân phiên
   (round-robin) giữa nhiều `CECEnv` độc lập, cùng ghi vào **một** replay buffer và
   cùng cập nhật **một** actor/critic — cho hiệu quả huấn luyện tương tự nhưng không
   dùng `multiprocessing`/`torch.distributed` thật sự (để chạy đơn giản trên 1 máy/1 GPU).
   Nếu bạn có nhiều GPU/máy, có thể mở rộng bằng `torch.multiprocessing` hoặc
   `torch.distributed` — kiến trúc replay buffer + actor/critic giữ nguyên.

2. **Action rời rạc trong DDPG**: bài báo dùng DDPG (vốn cho action liên tục) để
   *đồng thời* ra quyết định offload (rời rạc: chọn node) và băng thông (liên tục).
   Bài báo không nêu chi tiết cách rời rạc hóa. Ở đây mình dùng một đầu **softmax**
   cho lựa chọn node (lấy `argmax` lúc thực thi, nhưng vẫn khả vi để lan truyền
   gradient DDPG chuẩn như Algorithm 1) — đây là cách phổ biến trong literature
   offloading-RL để giữ được policy-gradient liên tục của DDPG.

3. **Topology mạng & bộ sinh DAG**: bài báo trích dẫn generator ngoài ([23], [28])
   không có source code công khai đầy đủ. `dag_generator.py` triển khai một bộ sinh
   DAG nhiều lớp (layer-by-layer) tiêu chuẩn, còn `edge_env.NetworkModel` dùng
   Erdos–Rényi (vá lại để đảm bảo liên thông) — tái tạo đúng các thuộc tính thống kê
   được mô tả (phân phối chuẩn cho tốc độ xử lý/băng thông, hệ số biến thiên 80%,
   thời gian release theo Poisson, v.v.) nhưng không phải bit-for-bit generator gốc.

4. **Testbed thực tế (Sec 6.2, Fig. 5–6)**: phần thí nghiệm trên thiết bị Jetson vật
   lý không thể tái tạo bằng mô phỏng — `evaluate.py` chỉ tái tạo phần **mô phỏng**
   (Sec 6.3–6.9, Fig. 7–10), là phần có thể chạy hoàn toàn bằng code.

=> **Kết luận**: bạn sẽ thấy đúng *xu hướng định tính* của bài báo (ADPRL/ADPRL-EE
vượt trội Random/LE/Greedy/DQN+FCFS về ACT và EC, đặc biệt khi số subtask/tác vụ
tăng hoặc băng thông/tốc độ xử lý thấp), nhưng **các con số phần trăm chính xác
(ví dụ "36.9% thấp hơn Random") sẽ không khớp tuyệt đối** với bài báo, vì phụ thuộc
vào các chi tiết triển khai không được công bố.

---

## 6. Mẹo tinh chỉnh / mở rộng

- **Tăng chất lượng ADPRL**: tăng `--episodes`, tăng `hidden` trong `networks.py`
  (mặc định 128), hoặc thêm decay cho `exploration_sigma` trong `ddpg_agent.py`
  theo thời gian huấn luyện (hiện tại cố định 0.3).
- **Reward shaping**: `edge_env.CECEnv._step_reward` hiện dùng reward "dense" theo
  từng subtask để agent học nhanh hơn (thay vì chỉ có reward cuối episode như
  Eq. 23 nguyên bản) — đúng tinh thần "dependency-aware reward mechanism" của Sec 5.4
  (khuyến khích agent xử lý sớm các subtask "nghẽn cổ chai" phụ thuộc). Bạn có thể
  chỉnh trọng số/hàm chuẩn hoá trong hàm này để thử nghiệm reward khác.
- **So sánh công bằng hơn với DQN+FCFS**: tăng `--dqn-warmup` trong `evaluate.py`
  (mặc định 50 episode) để DQN hội tụ tốt hơn trước khi đo — bài báo train tới hội tụ
  (~10.000 sự kiện) trước khi so sánh.
- **Lưu & tiếp tục huấn luyện**: `ADPRLAgent.save()/load()` chỉ lưu actor+critic
  (không lưu optimizer state/replay buffer) — nếu cần resume chính xác, có thể mở
  rộng `save()`/`load()` để lưu thêm `self.actor_opt.state_dict()` v.v.

---

## 7. Chạy nhanh kiểm tra (không cần GPU, vài giây)

```bash
python3 train.py --objective LO --episodes 20 --tasks-per-episode 5 \
    --edge-nodes 8 --workers 2 --device cpu --out runs/smoke_lo.pt

python3 evaluate.py --lo-model runs/smoke_lo.pt --out-dir results_smoke --quick
```

Nếu 2 lệnh trên chạy xong không lỗi và tạo ra file trong `runs/`, `results_smoke/`
thì môi trường của bạn đã sẵn sàng để chạy full training/evaluation trên GPU.
