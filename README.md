# TrajAgg Explorer

一个用于展示 **TrajAgg 轨迹相似度检索** 工作流程的中文学术 Demo。它不是参数调优平台，也不会在浏览器中训练模型。

> **当前状态：Prototype · example data。** 页面中的轨迹、轨迹 ID、Chebyshev 距离、预测相似度和 Hausdorff 距离均为本地静态示例，只用于展示交互逻辑，不能作为论文结果或本机复现实验结果引用。

## 启动

需要 Node.js 20.19 或更高版本。

```bash
cd ~/Desktop/demo页面
npm install
npm run dev
```

终端会给出本地访问地址（通常是 `http://localhost:5173`）。发布前可运行：

```bash
npm run build
```

## 固定学术配置

第一版严格固定以下展示配置，不提供自由参数调优：

| 项目 | 固定值 |
| --- | --- |
| 数据集 | Porto |
| 真实距离监督 | Hausdorff |
| 在线 embedding 检索距离 | Chebyshev |
| 网格单元 | 100 m |
| 双尺度融合系数 | μ = 0.5 |
| 训练模式 | Hybrid（embedding + pairwise） |

这体现的是 TrajAgg 的一个代表性工作流，而不是声称当前网页已严格完成论文的全部训练。

## 页面能展示什么

1. **查询轨迹选择**：左栏切换 Q-0001 至 Q-0005，GPS 与 100 m 网格预览、地图和结果会同步变化。
2. **Top-k 检索过程**：点击“检索 Top-k”会按“预处理 → TrajAgg 编码 → Chebyshev 排序 → 返回结果”显示短暂状态。
3. **路线对比**：中央区域呈现查询轨迹（蓝色）与 Top-k 候选（橙、绿、紫）；点击候选后会高亮该条路线。
4. **候选解释**：每张候选卡片的 **Why this candidate?** 会展开抽屉，说明 embedding 排名和 Hausdorff 真实距离的不同职责。
5. **Model Trace**：底部可展开区用三步解释当前查询：
   - Query & grid：GPS 与网格化双尺度输入；
   - Dual-scale encoder：GPS / Grid encoder 聚合为一个 embedding；
   - Chebyshev Top-k：query embedding 与轨迹 embedding 库比较、排序、返回候选。

页面明确不显示 Precision、Recall、NDCG，不伪造注意力热图。接入真实数据后，评测区域应使用作者代码对应的 **HR@1、HR@5、HR@10、HR@20、HR@50、R10@50**。

## 数据组织

```text
public/data/
├── index.json                 # 固定配置、可选择的查询轨迹列表、数据来源提示
└── cases/
    ├── Q-0001.json            # 一个查询案例
    ├── Q-0002.json
    └── ...
```

每个案例 JSON 包含：

- 查询轨迹的展示用 GPS / Mercator 坐标序列与 100 m 网格序列；
- Top-k 候选轨迹及其展示元数据；
- Chebyshev embedding 距离、预测相似度、可选 Hausdorff 真实距离；
- Model Trace 所需的 encoder / embedding 展示信息。

当前 JSON 是示例数据。真实导出时，应保留字段结构，另将每个数值的来源、checkpoint、数据切分与训练 epoch 记录在 `index.json` 的元数据中，以区分“论文正式结果”“本机预复现结果”和“互动案例数据”。

## 接入真实 TrajAgg 数据（第二阶段）

预留脚本在：

```text
scripts/export_trajagg_demo_data.py
```

它目前只验证未来导出所需输入并打印接入步骤，**不会输出伪造的真实模型结果**。未来应从 TrajAgg 作者实现导出：

- 训练完成的 TrajAgg checkpoint；
- Porto 库轨迹及其 ID、GPS / Mercator 序列、100 m grid 序列；
- 每条库轨迹的离线 embedding；
- 可选 Hausdorff 距离矩阵。

然后按以下顺序实际接入：复用作者的预处理 → `model.eval()` 编码查询与库 → Chebyshev 计算并排序 → 写入 `public/data/` JSON。前端只读取离线导出的 JSON，绝不在浏览器中运行 PyTorch 训练。

## 发布到 GitHub Pages

项目包含 [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)。推送到 GitHub 后：

1. 在 GitHub 新建仓库（例如 `trajagg-explorer`），将本文件夹内容提交并推送到 `main` 分支；
2. 在仓库 **Settings → Pages** 中将 Source 设为 **GitHub Actions**；
3. 之后每次推送 `main`，工作流会构建 `dist/` 并更新 GitHub Pages。

项目使用 **React + Vite + TypeScript**，并设置了相对资源路径，因此可作为 GitHub Pages 项目站点部署。

## 目录说明

| 路径 | 作用 |
| --- | --- |
| `src/App.tsx` | 主页面、检索状态、轨迹切换、Top-k、候选解释抽屉与 Model Trace 交互 |
| `src/styles.css` | 三栏学术 Demo 布局、路线可视化、响应式样式 |
| `public/data/` | 前端读取的静态案例数据；以后由离线导出替换 |
| `scripts/export_trajagg_demo_data.py` | 真实模型数据导出接口说明 |
| `.github/workflows/deploy.yml` | GitHub Pages 自动部署工作流 |

## 学术使用提醒

TrajAgg 的核心是将轨迹表示为 embedding 并进行快速检索。在线阶段按 Chebyshev embedding 距离返回 Top-k；Hausdorff 是离线监督 / 评测中使用的真实轨迹距离。两者不能混为同一个在线计算步骤。
