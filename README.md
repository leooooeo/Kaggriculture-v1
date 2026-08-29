# Kaggriculture 模仿学习 Agent

对**榜一（或任一强手）**的回放做行为克隆（Behavioral Cloning），训练一个读当前
observation 做反应的小型策略网络。相比 `new.py` 那种「固定脚本开局」，反应式策略
在 API 加入更多随机性后更稳——这正是本方案的核心动机。

## 为什么这样设计

每回合的动作是**复合 + 变长**的：`farmer`（1 个）+ `hands`（0～N 个雇工，各一个动作）
+ `market`（有序订单表）。直接让一个网络输出这整坨很难训。这里把它**因子化**成几个
可学习的头，全部从专家回放里学（不含任何手写策略）：

- **共享编码器**：10×10 棋盘多通道 CNN + 全局特征 MLP → 空间特征图 `S` + 全局上下文 `ctx`。
- **单位头**：对每个 unit 在其 `(x,y)` 处 gather `S`，配合 `ctx`，预测
  动作类型(18) + 作物(PLANT) + 物品/数量(PICKUP/PLACE)。天然支持雇工数量可变。
- **市场头**：从 `ctx` 预测「每个(订单类型,物品) 是否下单 + 数量分桶」、HIRE 数量、
  BUY_LAND。解码时按固定优先级组装成 ≤10 条的订单表。

模型仅 ~0.67M 参数（"小网络"）。全部代码见 `il/`。

## 目录

```
main.py               # Kaggle 提交入口: agent(obs, config)
il/spec.py            # 词表 / 动作表 / 归一化常量（训练与推理共用，防止漂移）
il/features.py        # obs -> (board, glob) numpy 特征（无需 torch）
il/action_codec.py    # 动作 <-> 因子化标签 的编解码（含合法订单表解码）
il/dataset.py         # 回放 JSON -> 训练分片 .npz
il/model.py           # PolicyNet（PyTorch，import 有 torch 保护）
il/train.py           # 行为克隆训练（掩码损失 + 类不平衡加权）
il/policy_agent.py    # 运行时：obs -> action；无权重/torch 时回退安全启发式
il/evaluate.py        # 在真实环境里对基线跑评估
```

## 在 Colab 上跑（你的主要流程）

```bash
# 1) clone 本库
!git clone -b claude/kaggriculture-farming-agent-1mhbgh <你的仓库URL> kagg && cd kagg

# 2) 安装依赖
!pip install -q numpy torch kaggle-environments

# 3) 放入回放：把榜一的 30 局回放 JSON 拷到 data/replays/
#    （用 kaggle CLI 拉：kaggle competitions replay <EPISODE_ID> -p data/replays）

# 4) 建训练分片。--team 填榜一的确切 TeamName（区分大小写）
!python -m il.dataset --replays data/replays --out data/shards --team "TOP1_TEAM_NAME"

# 5) 训练（Colab GPU 上把 epochs 调到 40~80）
!python -m il.train --shards data/shards --out models/policy.pt --epochs 60 --batch 512

# 6) 本地对战评估
!python -m il.evaluate --games 6 --opponent random
!python -m il.evaluate --games 4 --opponent new.py     # 需要仓库里的 new.py

# 7) 打包提交
!tar -czf submission.tar.gz main.py il/ models/policy.pt
!kaggle competitions submit kaggriculture -f submission.tar.gz -m "IL clone of TOP1 v1"
```

`--team` 省略时，脚本默认克隆**每局的赢家 seat**（"学赢家"）。要专注克隆某个人，
请传其确切 TeamName。

## 实测结论（重要，先读）

在本地用 One-For-All 的 30 局做过端到端验证，得到两条对训练预算很关键的结论：

1. **配对方向决定成败**。`dataset.py` 已修正为 `obs_{t-1} -> action_t`
   （kaggle-env 的 `steps[t].action` 作用于 `steps[t-1]` 才产生 `steps[t]`）。
   配错时模型全程 PASS；配对之后立刻学会开局爆发、雇工、喂养、种植、买卖的**节奏**。
2. **BC 的 loss 下降 ≠ 游戏分数上升**。16 轮 vs 40 轮：训练 loss 明显下降
   （type 1.19→0.74），但部署分数没升反微降，且更爱照搬专家高频动作（PICKUP）而脱离上下文。
   **纯 BC 克隆脚本型专家会在"差不多花光钱、勉强剩几百"附近触顶**——瓶颈不是训练时长，
   而是 (a) 协变量偏移（漏一次喂/浇 → 动物逃/作物成杂草 → 永久 off-distribution 级联崩溃），
   (b) 单一专家数据量，(c) 方法本身。

**因此，真正想上分的杠杆不是堆 epoch，而是：**
- 克隆**更反应式**的专家（榜一若不是固定脚本，会比 One-For-All 好克隆得多）；
- **数据广度**（多局、多状态）；
- **BC 之后接自对弈 / DAgger 纠偏**（在环境里跑当前策略，用专家或价值信号纠正 off-distribution 状态）。

## 已知要点 / 调参

- **类不平衡**：PASS 与「空市场」占绝大多数。训练已做逆频率类权重
  (`--pass_weight` 下压 PASS) 和市场正类 `pos_weight`。若 agent 仍塌缩到只 PASS/只移动，
  调大 `--pass_weight`（更激进：0.15）或增大 `--market_weight`，并加训 epoch。
- **数据量**：单一榜一 30 局 ≈ 2 万 step。够训出「节奏」，但 GPU 上多训。
  想更强可后续接**自对弈/DAgger**（在环境里跑当前策略，用专家/规则纠偏后回灌）。
- **推理阈值**：市场 present 用 sigmoid>0.5；如订单太保守可在 `policy_agent.py` 调低。

## 本地快速自检（无 GPU 也可）

```bash
pip install numpy torch kaggle-environments
python -m il.dataset --replays data/replays --out data/shards --team "One-For-All"
python -m il.train   --shards data/shards --out models/policy.pt --epochs 12
python -m il.evaluate --games 4 --opponent random
```
