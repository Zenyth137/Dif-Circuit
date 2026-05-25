# VLSI Macro Placement via Sequential RL with Imitation Learning

## 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                    训练流程 (三阶段)                       │
├───────────────┬──────────────────┬──────────────────────┤
│  B*-tree 打包  │  Imitation 预训练 │   PPO 微调 (+ ICM)   │
│  生成 expert   │   Behavior       │   强化学习精细化       │
│  演示数据      │   Cloning        │                      │
└───────┬───────┴────────┬─────────┴──────────┬───────────┘
        │                │                    │
        ▼                ▼                    ▼
   合法 placement   policy ≈ expert      policy 优化 HPWL
   (0 overlap)     loss = CE(action)    + overlap penalty
                                       + curiosity bonus
                                       + action masking
```

**推理阶段：** MDP 粗放置 → Diffusion 精修（去 overlap + 微调 HPWL）

---

## 1. 问题建模：Sequential MDP

将宏模块放置建模为**序贯决策过程**，每步放置一个模块到网格位置：

- **状态** $s_t = [\text{density\_grid}, \text{module\_w}, \text{module\_h}, \text{graph\_embedding}]$
  - `density_grid`：$G \times G$ 网格，记录已放置模块的空间占用（经过 CNN 编码为特征向量）
  - `module_w, module_h`：当前待放置模块的尺寸
  - `graph_embedding`：Edge-GNN 编码的网表拓扑特征（全局图池化向量）
- **动作** $a_t$：选择网格位置 $(gx, gy) \in \{0,\dots,G-1\}^2$，$G=64$，共 $4096$ 个离散动作
- **奖励**（见第 4 节）

---

## 2. Policy Network 架构

```
┌──────────────────────────────────────────────────────┐
│                  PolicyNet (Actor-Critic)              │
├──────────────────────────────────────────────────────┤
│                                                        │
│  Edge-GNN Encoder (3 layers, 4 heads)                  │
│  ├─ 节点特征: [module_w, module_h]                      │
│  ├─ 边特征: pin weights                                │
│  └─ 输出: per-node embeddings + global pooling         │
│                                                        │
│  Density Encoder (CNN)                                 │
│  ├─ Conv2d(1→8, 3×3) → ReLU                           │
│  ├─ Conv2d(8→16, 3×3) → ReLU                          │
│  ├─ AdaptiveAvgPool2d(4×4) → Flatten                  │
│  └─ Linear → density_dim=32                            │
│                                                        │
│  State Projector                                       │
│     [w, h | global_emb | density_feat] → MLP → 128-d   │
│                                                        │
│     ┌───────────────┐  ┌───────────────┐               │
│     │  Policy Head   │  │  Value Head   │               │
│     │  (Factorized)  │  │  (MLP → 1)    │               │
│     │  row_logits    │  │  V(s) 标量     │               │
│     │  + col_logits  │  │               │               │
│     │  → (G,G) logits│  │               │               │
│     └───────┬───────┘  └───────────────┘               │
│             │                                           │
│      π(a|s) = Softmax(flat_logits)                     │
│      动作空间: G² = 4096                                │
└──────────────────────────────────────────────────────┘
```

**Policy Head 因子分解：** 将 $G^2$ 个动作分解为行+列两个独立分布，$G + G = 128$ 个参数代替 $G^2 = 4096$ 个，显著减少参数量并提高训练效率。

---

## 3. 三阶段训练流程

### 阶段一：B\*-tree 打包生成 Expert 示范

使用 **B\*-tree** 紧凑表示 + **轮廓线打包算法（Contour Packing）** 生成合法（无重叠）placement：

- B\*-tree 随机构建 → in-order 遍历 → 轮廓跟踪确定各模块的 y 坐标
- 左侧子节点置于父节点右侧（$x_{\text{left}} = x_{\text{parent}} + w_{\text{parent}}$）
- 右侧子节点置于父节点上方（$x_{\text{right}} = x_{\text{parent}}$），y 由轮廓最高点确定
- 保证 **0 重叠**，HPWL 约 5,000–6,000

生成 **2000 条轨迹**，每条按模块连通度排序（高连通度优先放置），转换为 sequential 动作序列：
$$ \text{trajectory} = \{(w_i, h_i, \text{target\_grid\_action}_i)\}_{i=1}^{N} $$

### 阶段二：Behavior Cloning 预训练

使用交叉熵损失训练 policy 模仿 expert 动作：

$$ \mathcal{L}_{\text{BC}} = -\sum_{t} \log \pi_\theta(a_t^{\text{expert}} \mid s_t) $$

- 预提取所有轨迹的 GNN + State 特征（GNN 每个网表运行一次，400K 个训练对）
- 50 epoch，batch size 64，Adam (lr=1e-3)
- 训练后 policy 在训练网表上 HPWL ≈ 1,800（vs SA 的 5,400）

### 阶段三：PPO + ICM 微调

在预训练 policy 基础上，用 PPO 继续优化，同时加入两个关键机制：

#### 3.1 Intrinsic Curiosity Module (ICM)

解决稀疏奖励问题——原始 reward 仅在最终步给出。ICM 提供**每步 dense 探索信号**：

- **Forward Dynamics:** $f(\phi(s_t), a_t) \rightarrow \hat{\phi}(s_{t+1})$
- **Intrinsic Reward:** $r_t^{\text{int}} = \eta \cdot \|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\|^2$
- 预测误差大 → 状态新颖 → 鼓励探索

ICM 与 policy **联合训练**（独立 optimizer），loss 包含 forward MSE + inverse cross-entropy。

#### 3.2 Action Masking

防止模块重叠：环境实时计算 legal/illegal 动作掩码：

- 对每个网格单元，检查放置当前模块是否会与已放置模块重叠（通过 density grid 的 footprint 检查）
- 被占据区域 → mask = 0（logit 设为 $-\infty$）
- **训练和推理**均使用 mask，确保 policy 在约束下学习

---

## 4. 奖励设计

### Per-Step Dense Reward（每步即时反馈）

$$ r_t^{\text{dense}} = -\big( w_{\text{hpwl}} \cdot \text{HPWL}_{\text{new}} + w_{\text{overlap}} \cdot \text{Overlap}_{\text{new}} \big) $$

- $\text{HPWL}_{\text{new}}$：当前模块使得新"完整"线网的半周长增量
- $\text{Overlap}_{\text{new}}$：当前模块与已放置模块的重叠面积

### Terminal Reward（最终全局评估）

$$ R_{\text{terminal}} = -\big( w_{\text{hpwl}} \cdot \text{HPWL}_{\text{full}} + w_{\text{congestion}} \cdot \text{Congestion} + w_{\text{overlap}} \cdot \text{Overlap}_{\text{full}} \big) $$

权重配置：$w_{\text{hpwl}}=1.0,\; w_{\text{congestion}}=0.5,\; w_{\text{overlap}}=500.0$

### Combined Reward for PPO

$$ R_{\text{total}} = \sum_t (r_t^{\text{dense}} + r_t^{\text{int}}) + R_{\text{terminal}} $$

---

## 5. Diffusion 精修

MDP 粗放置后，使用 **Denoising Diffusion Probabilistic Model** 进行精修：

- 输入：MDP 粗放置坐标 + 模块尺寸 + 网表拓扑（GNN 编码）
- 扩散过程：逐步去噪，同时施加 **能量引导（Energy Guidance）**：
  - HPWL 能量：$\mathcal{E}_{\text{hpwl}} = \sum_{\text{net}} \text{LSE}(x, y)$
  - Overlap 能量：$\mathcal{E}_{\text{overlap}} = \sum_{i,j} \text{ReLU}(\text{overlap}_{ij})$
- **Variance Alignment**：将离散网格的量化方差匹配到扩散噪声方差，确保细粒度调整

---

## 6. PPO 训练细节

| 超参数 | 值 |
|--------|-----|
| Learning rate | 3e-4 |
| Discount $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| Clip $\epsilon$ | 0.2 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| PPO epochs | 10 |
| Batch size | 64 |
| Episodes per iter | 8–16 |

| ICM 参数 | 值 |
|----------|-----|
| Curiosity $\eta$ | 0.5 |
| ICM lr | 1e-4 |
| ICM hidden dim | 256 |
| Forward loss weight | 1.0 |
| Inverse loss weight | 0.2 |

---

## 7. 关键设计决策与经验

1. **为何需要 Imitation 预训练：** 从零开始 RL 探索在 200 步 × 4096 动作空间中几乎不可能找到好的 placement——实验中纯 RL 在 100 轮后 HPWL 仍为 ~1,060,000（随机水平）。Imitation 提供好的初始策略。

2. **为何 Action Mask 关键：** 不加 mask 时，policy 学会将所有模块堆叠在同一格（HPWL≈0, Overlap=100%）——典型的 reward hacking。Mask 强制合法 placement。

3. **Dense Reward 的作用：** 纯 terminal reward 导致 200 步中 199 步无信号。Dense reward（增量 HPWL + overlap）提供即时反馈，改善 credit assignment。

4. **ICM 的局限性：** ICM 提供探索信号但不能直接指导 HPWL 优化——agent 学会探索新状态但不一定降低 HPWL。与 Dense Reward 结合使用。

5. **B\*-tree 打包 vs SA：** B\*-tree 轮廓打包产生 0-overlap 合法 placement（HPWL ~5,000），速度远超 SA（0.008s vs 1.9s），适合大规模生成 expert 数据。
