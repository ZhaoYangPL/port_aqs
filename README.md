# PORT-AQS：Stage 1 RPM-Aware Training-Free Online Routing

这是重构后的 Stage 1 可运行实现。它研究：

> 在只调用可并发外部 API、完全不使用调用完成反馈时，静态 task-conditioned 画像与实时 RPM/TPM admission state 能否帮助 Router 降低端到端 SLO violation。

端到端延时统一定义为：

```text
route time + client quota admission wait + black-box API response latency
```

外部 API response latency 不拆成“供应商排队 + service time”。当前合成 MVP 将 `routing_latency` 固定为 0；它不调用任何真实 API。

## 本实现做了什么

- 在 RouterBench historical split 上建立静态 TF-IDF 加权 kNN 画像；
- 同时预测 quality、monetary cost、output-token demand 和黑盒 latency 分布；
- 为每个匿名 endpoint 维护 RPM request-permit bucket 和 TPM LLM-token bucket；
- 用 `max(rpm_wait, tpm_wait)` 作为客户端 joint admission wait；
- 在 calibration split 上给定成本预算 $B^C$，解一次 proxy LP，得到并冻结 `beta`、`gamma_rpm` 和 `gamma_tpm`；
- 固定 `beta/gamma` 后，只在 calibration split 上 replay 候选 `lambda`，得到并冻结 `lambda`；
- streaming 中一次性路由、FIFO reservation、无 retry，最后 drain；
- 输出逐请求候选诊断、双 quota 状态和完整延时/质量/成本指标。

## 本实现没有做什么

Stage 1 完全没有 execution completion feedback：

- 没有 latency health EWMA；
- 没有 `update_on_completion` 或 completion 后 latency CDF 更新；
- 没有 verifier-grounded quality update；
- 没有按实际输出 token 回补/扣减 TPM reservation；
- 没有 endpoint slowdown 适应、incident detection 或 recovery 指标；
- 没有读取未选 arm、未完成调用、未来 trace 或 streaming full outcome matrix；
- 没有本地 serving、客户端全局并发上限、retry、hedging 或多轮 Agent routing。

completion 只用于 evaluator 记录真实 outcome、计算指标和 drain，不会改变未来 route。

## 快速运行

在本目录执行：

```powershell
python -m pip install -e .
python -m port_aqs.experiment --config configs/mvp.yaml --smoke
```

若当前环境已经具备依赖，可省略安装。Smoke 使用 400/100/120 条 historical/calibration/streaming 数据、seed 0，以及 joint、RPM-only、TPM-only 三个条件；7 个公开策略共运行 21 个 run。它只检查数据、calibration、双 quota、仿真、drain、指标与 trace 管线，不能据此声称方法优于 baseline。

完整矩阵：

```powershell
python -m port_aqs.experiment --config configs/stage1_stable_rpm.yaml
```

常用覆盖参数：

```powershell
python -m port_aqs.experiment --list-policies
python -m port_aqs.experiment --config configs/stage1_stable_rpm.yaml --policies rpm_aware static_latency
python -m port_aqs.experiment --config configs/stage1_stable_rpm.yaml --conditions stable_rho_050 stable_rho_070
python -m port_aqs.experiment --config configs/stage1_stable_rpm.yaml --seeds 0 1 --no-traces
python -m port_aqs.experiment --config configs/stage1_stable_rpm.yaml --output results/custom
```

## 数据与静态画像

默认数据是 `../../PORT/data/routerbench_0shot.pkl`。加载器找到同时具有 quality、`|total_cost` 和 `|model_response` 的 11 个模型列，并立即匿名化为 `arm_00 ... arm_10`。

原数据 36,497 行。按 prompt SHA-256 去掉 16 个重复项，再按 `eval_name` 分层划分：

| Split | 数量 | 用途 |
|---|---:|---|
| historical | 26,481 | 固定 kNN 画像 |
| calibration | 500 | 按固定 quota load 校准并冻结参数 |
| streaming | 9,500 | 在线 replay 与离线评价 |

RouterBench 没有真实 API latency。因此代码先用：

```text
a_i + 0.001 * input_tokens + output_tokens / v_i
```

再乘 mean-one log-normal shock 和少量 tail spike，生成匿名 task–arm 黑盒 latency。Stage 1 没有随 dispatch 时间变化的 health multiplier。

当前 MVP 用 `ceil(字符数 / 4)`（至少 1）作为输入/输出 token proxy。这不是 tokenizer-accurate billing；真实 API 阶段必须替换为相应 tokenizer 或 API `usage`。

对于当前 prompt，TF-IDF cosine kNN 找到 historical 邻居，同一组 inverse-distance 权重用于：

- 加权 quality；
- 加权 monetary cost；
- 加权预测 output token；
- 加权经验 latency CDF 和 p95。

所有 streaming 实际 outcome 只对 evaluator 可见，不能写回画像。

## RPM permit、TPM token 与金钱成本

这三个概念不能混用：

| 资源 | 每个请求的用量 | 作用 |
|---|---:|---|
| RPM request permit | 固定 1 | 硬 admission 约束 |
| TPM LLM token | `input_tokens + predicted_output_tokens` | 硬的预测 reservation 约束 |
| Monetary cost | 随 task/arm 变化 | 软效用惩罚和评价指标 |

真实输出在 dispatch 前未知，所以 TPM 使用：

```text
predicted_token_demand = known_input_tokens + knn_predicted_output_tokens
```

Stage 1 的硬约束准确地说是“严格执行预测 reservation”。实际输出 token 只供离线报告预测误差与实际 consumption，不在 completion 后 reconcile。若真实系统需要保守保证，可以用 `max_tokens` reservation；actual-use reconciliation 属于 Stage 2。

## 双 Quota Admission

每个 endpoint 独立配置 RPM/TPM refill rate 和 bucket capacity。新请求排在该 endpoint 的 reservation FIFO 末尾，同时预览：

```text
rpm_wait = wait until one request permit is available
tpm_wait = wait until predicted_token_demand is available
admission_wait = max(rpm_wait, tpm_wait)
```

RPM-only 条件下，每个请求固定消耗一个 permit，实现用一个可为负的 RPM virtual balance 直接计算等待，preview/commit 不回放 pending queue。joint RPM/TPM 条件下，TPM demand 随请求变化，而且一个 bucket 等待时另一个 bucket 的 refill 可能因容量上限被浪费，因此仍按 FIFO 精确回放已承诺 reservation。被设为无限的资源视为 disabled，其 ready wait 始终为 0。

`preview(timestamp, token_demand)` 无副作用；`commit(timestamp, token_demand, request_id)` 只修改被选 endpoint。dispatch 同时消费 1 个 permit 和预测 LLM token。API completion 不归还任何资源，也不改变 quota state。

如果 TPM bucket capacity 小于单请求 reservation，配置/commit 会失败；本阶段不会把一个请求静默拆分。

## Calibration LP、静态 SLO Risk 与打分

Stage 1 仍然是 training-free。这里的 calibration 不是训练一个模型，也不是在 streaming 阶段利用反馈调参。每个预先声明的 quota-load condition 都使用同一份 500 条 calibration split：先解静态 proxy LP 得到该 condition 的资源对偶价格，再 replay 选择该 condition 的 lambda。

给定 calibration 任务集合 $\mathcal C$，对每个任务 $m$ 和端点 $i$ 已有 kNN 预测：

- 预测质量 $\widehat u_{im}$；
- 归一化预测成本 $\widetilde c_{im}$；
- 预测 token demand，仅在 TPM 条件下使用。

正式 RPM-only Stage 1 的 calibration proxy LP 是：

```text
max_y     sum_{m,i} y_{mi} * predicted_quality_{mi}
s.t.      sum_i y_{mi} = 1                         for every task m
          sum_{m,i} y_{mi} * normalized_cost_{mi} <= B^C
          sum_m y_{mi} <= K_i + r_i H              for every endpoint i
          y_{mi} >= 0
```

其中：

- $B^C$ 是 calibration 阶段预先给定的总归一化成本预算；
- $K_i$ 是端点 $i$ 的 RPM bucket 容量；
- $r_i=\mathrm{RPM}_i/60$ 是端点 $i$ 每秒恢复的 request permit 数；
- $H$ 是 calibration workload 的 nominal horizon。

LP 的 cost 约束对偶变量就是 `beta`，RPM capacity 约束对偶变量就是 `gamma_rpm`。因此 `beta` 和 `gamma` 都来自同一个 PORT-style 小样本 LP，而不是通过 streaming 结果或 replay grid search 调出来。

`lambda` 不从这个 LP 中求出，因为 SLO risk 包含实时 admission wait，而 admission wait 取决于之前的在线路由动作，不是静态 task-arm 属性。实现会在 `beta/gamma` 固定后，只在 calibration split 上对候选 `lambda_grid` 做顺序 replay，并用预先写入配置的 selection utility 选择 `lambda`：

```text
quality_mean
- selection_cost_weight * normalized_cost_mean
- selection_slo_weight * slo_violation_rate
```

这个 replay 仍然发生在 streaming 之前，不读取 streaming 结果，也不使用 completion feedback。

对 deadline `D`：

```text
risk = 1                                  if D <= admission_wait
risk = 1 - F0(D - admission_wait | task) otherwise
```

路由分数：

```text
quality
- beta * normalized_monetary_cost
- gamma_rpm
- gamma_tpm * predicted_token_demand
- lambda * predicted_slo_risk
```

`gamma_rpm` 与 `gamma_tpm` 只是长期稀缺性先验；真正执行 rolling quota 的是两个 bucket。它们不是 PORT 原预算模型中同名变量的直接复现，也不继承其理论保证。

`beta`、`gamma` 和 `lambda` 在每个固定 quota-load condition 的 streaming 开始前冻结，并在该 condition 的全部策略和 10 个 seed 间保持相同。不同 load 会各自完整重做 LP 与 lambda replay。后续如果研究 SLO 偏好的敏感性，应改变 `lambda_grid` 或 selection weights 后重跑 calibration，而不能看完 streaming 结果再挑。

## 策略

核心策略：

- `quality_cost`：不使用 latency risk，即 `lambda=0`；
- `static_latency`：使用冻结 API response CDF，但 route 时令 admission wait 为 0；
- `rpm_aware`：主方法，使用静态 CDF 与实时 RPM/TPM joint wait；
- `available`：优先从 `admission_wait=0` 的 endpoint 中按静态效用选择，否则先最小化 wait。

消融策略：

- `rpm_aware_no_gamma`：去掉长期 quota shadow price，只看实时风险项；
- `rpm_aware_lambda_0`：令 `lambda=0`，用于验证退化到 quality-cost 路由；
- `static_latency_no_gamma`：去掉静态延时版本中的长期 quota shadow price。

诊断参考包括 `random`、`best_quality` 和 `min_latency_risk`。`admission`、`quota_risk`、`static_risk` 和 `min_risk` 只保留为兼容旧命令的 alias；它们不是新配置或论文表格中的正式策略名。Stage 1 没有 `full/dynamic_health` 或 perfect-health oracle。

## 实验场景

完整配置围绕稳定 endpoint 的 quota 条件展开：

```text
cal_mean_demand_i = endpoint i 在 calibration 上的平均 predicted token demand
c_i = min(rpm_i / 60, (tpm_i / 60) / cal_mean_demand_i)
rho_eq = arrival_rate / sum_i(c_i)
```

YAML 条件名中的 `rho` 都指这个冻结的 nominal request-equivalent load `rho_eq`，不是根据某个测试策略或实际 test output 事后计算的 load。RPM-only 将 TPM 设为无限，TPM-only 将 RPM 设为无限。

- joint RPM/TPM load sweep：`rho_eq = 0.3/0.5/0.7/0.9/0.95`；
- RPM-only pressure；
- TPM-only pressure；
- joint RPM/TPM 的有限 `0.5 -> 1.2 -> 0.5` burst；
- 至少 5 个异构 RPM/TPM 固定 permutation；
- 10 个共同随机种子和 paired confidence interval。

RPM-only 或 TPM-only 的单资源 burst 是可选扩展，不在默认完整矩阵中。持续 load 大于 1 只作为有限 burst，不作为稳态。配置中没有 slowdown/incident/health 条件。

## 输出

默认输出目录由 YAML 决定；`--smoke` 默认写入 `results/smoke/`。主要文件：

- `metadata.json`：匿名 arms、去重 split、token/latency 合成配置和冻结参数；
- `resolved_config.json`：应用 smoke/CLI override 后的最终配置；
- `split_manifest.csv`：task/hash/split，用于检查泄漏；
- `calibration_trials.csv`：逐 condition 记录固定 LP dual 后不同 `lambda` 的 calibration replay，以及各 condition 选中的 `lambda`；
- `condition_parameters.json`：每个条件的 RPM/TPM 配置、$B^C$、capacity horizon、冻结 beta、lambda 与双 gamma；
- `run_metrics.csv`：每个 condition/seed/policy 的完整标量；
- `aggregate_metrics.csv`：condition/policy 聚合结果；
- `paired_confidence_intervals.csv`：共同 seed 下主方法相对 baselines 的配对区间；
- `traces/*.pkl.gz`：逐请求完整 trace，可用 `--no-traces` 关闭。稳定接口中，candidate diagnostics 保存每个 arm 的总 predicted token demand、quota snapshot/wait、risk、p95 和 score；顶层 selected 行保存 predicted/actual output token 和 evaluator outcome。

主要指标包括 quality、cost、RPM/TPM/joint wait 的 mean/p95/p99、两种 quota binding rate、API response、E2E mean/p95/p99、SLO violation、CVaR95、逐 arm route share、routing HHI 和 risk calibration。RPM/TPM utilization 统一限定在首个 arrival 到末个 arrival 的共同观察窗口；后续 drain dispatch 不扩大分母，也不计入窗口内消费。Token 部分另报 prediction bias/MAE/RMSE/MAPE、reservation bias/MAE、`token_reservation_coverage_rate`、`token_underreservation_rate`、`actual_token_demand_total/mean`、基于预测 reservation 的 TPM utilization，以及只供 evaluator 计算的 `actual_tpm_utilization__arm/mean`。没有 incident recovery 指标。

## 可视化分析

可视化入口默认只读取小型 `run_metrics.csv`，自动排除 `available`，不会批量载入体积很大的逐请求 trace：

```powershell
python -m port_aqs.visualize results/stage1_stable_rpm/rho_030
```

默认输出到输入目录的 `analysis/`，包括：

- `analysis_report.html`：浏览器中按推荐顺序阅读的总报告；
- `analysis_report.pdf`：包含全部主图的单文件报告；
- `00_scorecard.png`：六个决策指标的精确值与条件内排名；
- `01_tradeoffs.png`：质量、SLO 与每千请求成本的 seed-level 权衡；
- `02_paired_effects.png`：主方法相对核心 baseline 的共同-seed 配对区间，所有坐标统一为“向右更好”；
- `03_seed_stability.png`：10 个共同随机种子的稳定性；
- `04_routing_mechanism.png`：route share、RPM utilization、HHI 与 binding rate；
- `analysis_summary.csv` 和 `paired_effects.csv`：只保留论文判读所需的精简数值表。

只有明确需要逐请求动态诊断时才读取某一个 seed 的 trace；实现会逐策略处理，不会一次拼接全部 trace：

```powershell
python -m port_aqs.visualize results/stage1_stable_rpm/rho_030 --trace-seed 0
```

后续多个 rho 目录可以一次传入。工具会额外生成 load-sweep 图；不同目录不得包含重复的 `condition/seed/policy`：

```powershell
python -m port_aqs.visualize `
  results/stage1_stable_rpm/rho_030 `
  results/stage1_stable_rpm/rho_050 `
  --output results/stage1_stable_rpm/analysis_combined
```

图中的 confidence interval 以 seed 为独立统计单位，而不是把 9,500 个请求当成独立样本。报告不会自动宣布“Stage 1 成功”：必须先在实验协议中声明最大允许质量下降和最大允许成本上升，才可以对总体权衡下通过/失败结论。

## 代码结构

```text
port_aqs/
  data.py        # RouterBench 匿名化、去重分层切分、静态 TF-IDF kNN
  synthetic.py   # stable latency/token potential outcomes 与 arrival
  quota.py       # RPM/TPM 联合 FIFO preview/commit
  latency.py     # 冻结的加权经验 latency CDF/quantile
  router.py      # quality-cost-dual-scarcity-SLO 一次性打分
  simulator.py   # 无反馈 replay、dispatch/completion drain
  metrics.py     # 双 quota、SLO、token-demand 与配对统计
  experiment.py  # YAML/CLI、calibration、实验矩阵和输出
  visualize.py   # 精简表、配对分析、静态报告与可选 trace 诊断
```

## 验证

```powershell
python -m unittest discover -s tests -v
python -m compileall -q port_aqs tests
```

重点验证：

- RPM/TPM conservation、FIFO、无负余额；
- completion 不改变 quota/画像/路由；
- route 不读取 streaming 实际输出 token 或实际 latency；
- `lambda=0` 严格退化为 `quality_cost`；
- 双 quota 无限时 `rpm_aware` 退化为 `static_latency`；
- TPM 充足时退化为 RPM-only，RPM 充足时退化为 TPM-only；
- 所有已 commit 请求在最终 arrival 后完整 drain。

## 后续阶段

- Stage 2A 才首次加入 selected-completion latency health 和可选 token reconciliation；
- Stage 2B 加入 verifier-grounded quality feedback；
- Stage 3 显式处理 selective/delayed feedback、探索和安全约束；
- Stage 4 处理 endpoint/model/version/pricing/RPM/TPM 配置变化。

详细研究定义见上一级目录的 `Stage1_PORT_Latency_Research_Guide.md` 和 `研究方向.md`。
