# gpt-oss-20b 中文结构化输出后训练实验方案

## 1. 项目目标

本项目聚焦一个清晰且工程上自然的目标：

**让 gpt-oss-20b 能够理解中文用户输入，并稳定输出正确的结构化结果 / function call。**

这里的重点不是“输出必须是中文自然语言”，而是：

- 能看懂中文指令
- 能抽取正确意图与参数
- 能输出合法 JSON / schema-constrained 结果
- 能在后训练后进一步提升结构化正确率与约束遵循能力

项目最终分为两个阶段：

1. **SFT 阶段**：先完成中文理解与结构化输出适配
2. **RLVR-lite 阶段**：再通过 verifier-guided RL 对结构化正确性做进一步优化

后续若时间允许，再将同一 recipe 扩展到 `gpt-oss-120b`，作为 100B+ MoE 模型上的规模验证与工程 profiling 扩展。

---

## 2. 总体路线

### 阶段 A：gpt-oss-20b 主线实验
- 中文 instruction + 中文结构化输出 SFT
- 中文结构化任务上的 RLVR-lite

### 阶段 B：扩展到 gpt-oss-120b
- 迁移同样的数据格式与训练 recipe
- 做短程验证
- 对比 20b 与 120b 的吞吐、显存、延迟、输出长度与结构化稳定性

本实验的主成果以 **20b 路线** 为主；120b 扩展用于补充“100B+ 模型参与”的工程证据。

---

## 3. 任务定义

### 3.1 核心任务

将中文输入映射到结构化输出，形式包括：

- JSON
- function calling
- schema-constrained arguments

不做开放式中文聊天作为核心任务。

### 3.2 示例

输入：

> 星期五早上九点叫醒我

目标输出：

```json
{
  "name": "alarm_set",
  "arguments": {
    "date": "星期五",
    "time": "早上九点"
  }
}
```

### 3.3 项目实质

本项目本质上是：

**中文语义理解 + 意图识别 + 槽位抽取 + 结构化执行接口预测**

这一定义天然适合：
- SFT
- verifier-based reward
- RLVR-lite

---

## 4. 数据方案

本项目不依赖自建数据，优先使用公开数据。

### 4.1 数据集一：COIG-CQIA
用途：
- 中文 instruction 理解打底
- 中文约束遵循
- 基础中文指令跟随能力增强

建议用途：
- 作为 SFT 的辅助数据
- 不进入 RL 阶段

### 4.2 数据集二：MASSIVE（zh-CN 子集）
用途：
- 主任务数据
- 中文 utterance 到结构化输出/function call 的映射

MASSIVE 样本关键字段包括：
- `utt`：原始中文输入
- `intent`：目标意图
- `annot_utt`：槽位标注后的 utterance
- `scenario`：场景类别
- `partition`：train/dev/test

示例：
- `utt`: 星期五早上九点叫醒我
- `intent`: alarm_set
- `annot_utt`: [date : 星期五] [time : 九点] 叫醒我

可转成：

```json
{
  "name": "alarm_set",
  "arguments": {
    "date": "星期五",
    "time": "九点"
  }
}
```

### 4.3 数据转换原则

MASSIVE 不直接拿来训练，而是统一转换成固定格式。

建议统一为 function calling 风格：

```json
{
  "name": "<intent>",
  "arguments": {
    "<slot_1>": "<value_1>",
    "<slot_2>": "<value_2>"
  }
}
```

### 4.4 数据配比建议（SFT）
初始建议：

- MASSIVE zh-CN 结构化改写版：70%
- COIG-CQIA：30%

原因：
- MASSIVE 是主任务数据
- COIG 只负责补中文 instruction 理解

---

## 5. 数据处理细节

### 5.1 MASSIVE 槽位解析

从 `annot_utt` 中解析槽位，模式类似：

```text
[slot_name : slot_value]
```

解析后得到 `arguments` 字典。

例如：

```text
[date : 星期五] [time : 九点] 叫醒我
```

解析为：

```json
{
  "date": "星期五",
  "time": "九点"
}
```

### 5.2 最终训练样本格式

建议先落成标准 chat/SFT 样本格式，例如：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "请将下面的中文请求转换为函数调用：\n星期五早上九点叫醒我"
    },
    {
      "role": "assistant",
      "content": "{\"name\": \"alarm_set\", \"arguments\": {\"date\": \"星期五\", \"time\": \"九点\"}}"
    }
  ]
}
```

之后再根据 gpt-oss 所需格式转成对应 chat template / Harmony-compatible 训练格式。

### 5.3 输出约束

输出不要求自然语言中文。  
结构化任务中更推荐：

- `name` 保持英文 intent 名
- `arguments` 的 key 保持标准英文 slot 名
- value 使用原中文值或标准化值

---

## 6. SFT 实验方案

### 6.1 目标
SFT 阶段解决两个问题：

1. 模型能看懂中文输入
2. 模型能稳定输出正确结构化结果

### 6.2 SFT 训练内容
- COIG-CQIA：中文 instruction 跟随
- MASSIVE zh-CN 改写版：中文 function calling / structured output

### 6.3 SFT 产出
SFT 后应具备：
- 基本中文理解能力
- intent 识别能力
- slot 抽取能力
- 结构化输出格式稳定性

### 6.4 SFT 基线
第一版建议至少做以下基线：

1. **仅 MASSIVE zh-CN**
2. **MASSIVE zh-CN + COIG-CQIA**

用于验证 COIG 是否真的对中文 instruction 理解有帮助。

---

## 7. RLVR-lite 实验方案

## 7.1 为什么选择 RLVR-lite

本项目不做开放式 RLHF，也不先做 reward model。  
原因：

- 结构化任务天然可验证
- verifier 易于编写
- reward 更可靠
- 更适合短周期落地

因此选择：

**verifier-guided RLVR-lite**

### 7.2 RL 仅作用于哪些数据
RL 阶段只使用：

- MASSIVE zh-CN 结构化改写子集

不把 COIG 引入 RL 阶段。

### 7.3 rollout 形式
对每个输入采样多个候选，例如：

- 每个样本采样 `K=4` 个候选输出

### 7.4 reward 设计

定义总 reward：

```math
r = w_1 r_{json} + w_2 r_{intent} + w_3 r_{slot} + w_4 r_{constraint}
```

其中：

#### (1) JSON 合法性
- `r_json ∈ {0,1}`
- 能 parse = 1
- 不能 parse = 0

#### (2) intent 正确性
- `r_intent ∈ {0,1}`
- `name` 是否等于 gold intent

#### (3) slot / argument 正确性
- `r_slot ∈ [0,1]`
- 用正确参数比例表示，例如：

```math
r_{slot} = \frac{\#correct\_slots}{\#required\_slots}
```

#### (4) 约束满足度
- `r_constraint ∈ [0,1]`
- 是否只输出 JSON
- 是否附加了多余解释
- 是否缺失必填字段
- 是否有多余字段

### 7.5 推荐权重
初始可尝试：

```text
w_json = 0.2
w_intent = 0.3
w_slot = 0.4
w_constraint = 0.1
```

后续可调。

### 7.6 RL 更新方式
优先选择简单、稳定的 group-based RL：

- 同一输入采样 K 个候选
- verifier 逐个打分
- 组内平均 reward 作为 baseline
- 做相对优势更新

若工程上已有 PPO / GRPO / RLOO 框架，也可根据现成工具选择最稳实现。

### 7.7 RL 阶段目标
RLVR-lite 主要提升：
- JSON parse rate
- intent accuracy
- slot correctness
- full-call exact match
- 格式遵循率

---

## 8. 评测方案

### 8.1 主评测指标

#### (1) Intent Accuracy
预测的 `name` 是否正确

#### (2) Slot F1 / Argument Accuracy
参数预测正确率

#### (3) JSON Parse Rate
输出是否是合法 JSON

#### (4) Schema / Required Field Pass Rate
字段是否齐全、类型是否正确

#### (5) Full-call Exact Match
整个 function call 是否完全正确

### 8.2 辅助评测指标

#### (1) 输出长度
平均输出 token 数

#### (2) 多余解释比例
明明要求只输出 JSON，但模型额外输出自然语言说明的比例

#### (3) 非法字段比例
输出中存在 schema 外字段的比例

### 8.3 对比实验
至少做以下对比：

1. Base gpt-oss-20b
2. 20b + SFT (MASSIVE only)
3. 20b + SFT (MASSIVE + COIG)
4. 20b + SFT + RLVR-lite

---

## 9. 工程与系统分析

除了效果指标，还要保留工程侧分析。

### 9.1 训练侧
记录：
- tokens/sec
- step time
- max memory
- checkpoint 时间
- 训练稳定性（OOM / NaN / hang）

### 9.2 推理侧
记录：
- 平均输出长度
- P50 / P95 latency
- JSON 合法率
- full-call exact match

### 9.3 对 120b 扩展时重点看
若扩到 gpt-oss-120b，重点不是完整复刻所有实验，而是分析：

- 同样 recipe 是否可迁移
- 吞吐下降多少
- 显存占用变化
- 输出长度变化
- structured output 稳定性收益是否值得其成本

---

## 10. 扩展到 gpt-oss-120b 的计划

### 10.1 扩展目标
不是另起一个新项目，而是回答：

> 同样的中文 structured-output 后训练方案，扩展到 100B+ MoE 后效果与成本如何变化？

### 10.2 建议做法
在 20b 路线跑通后，再对 120b 做：

- 短程 SFT 验证
- 少量 RLVR-lite 验证（可选）
- 系统 profiling
- 20b vs 120b 对照

### 10.3 120b 的项目价值
这一阶段用于保留：
- 100B+ MoE 模型参与
- 大模型后训练/推理系统经验
- 规模扩展的工程 trade-off 分析

---

## 11. 风险点与规避

### 风险 1：只做 SFT，项目显得偏弱
规避：
- 必须增加 RLVR-lite 第二阶段

### 风险 2：reward hack
现象：
- 模型学会输出空 JSON 或极短结果骗分

规避：
- reward 不能只看 parse
- 必须同时检查 intent / slot / required fields

### 风险 3：COIG 引入与主任务不一致的自然语言输出
规避：
- COIG 只做适量混入
- structured-output 主任务仍以 MASSIVE 为核心

### 风险 4：120b 扩展过重
规避：
- 120b 只做短程验证与 profile
- 不把 120b 作为主交付依赖

---

## 12. 最终项目表述（简历/汇报可用）

### 中文版
在 gpt-oss-20b 上构建中文 instruction 与 structured-output 后训练流水线，使用 COIG-CQIA 与 MASSIVE 中文子集完成中文理解到 function calling 的 SFT 适配，并进一步通过 verifier-guided RLVR-lite 优化 JSON/schema 合法率、intent/slot 正确率与约束满足率；随后将同一 recipe 扩展到 gpt-oss-120b，分析 100B+ MoE 模型上的吞吐、显存、延迟与效果 trade-off。

### 英文版
Built a Chinese instruction and structured-output post-training pipeline on gpt-oss-20b using COIG-CQIA and the zh-CN subset of MASSIVE for supervised adaptation from Chinese utterances to function-calling outputs, then improved JSON/schema validity, intent accuracy, slot extraction, and constraint satisfaction with verifier-guided RLVR-lite; further scaled the same recipe to gpt-oss-120b to study quality, throughput, memory, and latency trade-offs on a 100B+ MoE model.

---

## 13. 执行顺序建议

### Week 1
- 跑通 gpt-oss-20b 训练环境
- 下载并检查 COIG-CQIA 与 MASSIVE
- 完成 MASSIVE zh-CN 转 structured-output 脚本
- 做小规模 SFT smoke test

### Week 2
- 正式跑 20b SFT
- 建立 held-out eval
- 比较 MASSIVE-only vs MASSIVE+COIG

### Week 3
- 实现 verifier
- 跑 RLVR-lite
- 比较 SFT vs SFT+RLVR-lite

### Week 4
- 补 profile / latency / memory 分析
- 整理图表与报告
- 视情况开始 120b 短程扩展

---

## 14. 最终定稿

**主项目：**
- gpt-oss-20b 中文 structured-output/function-calling 后训练
- SFT + RLVR-lite

**扩展项目：**
- 将同一 recipe 扩展到 gpt-oss-120b
- 做 100B+ MoE 模型上的短程验证与工程分析

这就是当前最自然、最完整、最不拧巴的实验方案。
