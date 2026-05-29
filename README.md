# Aurum Matrix - 黄金当前价位评估工具

Aurum Matrix 是一个基于 **Python 3.11 + FastAPI** 的黄金当前价位评估工具。项目目标是评估当前黄金价格是否适合配置或交易，**不做确定性价格预测，不提供收益承诺**。

当前版本是第一版骨架：所有数据均来自 deterministic mock 数据，暂不接入真实行情、DXY 或 FRED API。

## 核心原则

1. 所有关键数值必须来自数据源或计算结果。
2. LLM 不允许自行编造任何价格、指标、仓位或收益。
3. 数据缺失时必须输出 `数据不足`。
4. 不提供收益承诺或确定性价格预测。
5. 输出必须包含风险提示。
6. 每个模块都要有类型注解和单元测试。

## 项目结构

```text
.
├── data_sources/       # 黄金价格、DXY、FRED 宏观数据源；当前为 mock 数据，含 SQLite 缓存骨架
├── indicators/         # MA、MACD、RSI、ATR 等技术指标
├── scoring/            # 宏观分、技术分、资金流分、总分
├── decision_engine/    # 输出 增配/持有/观望/减配/回避/数据不足
├── llm/                # 可选中文报告生成；当前为非 LLM 占位实现
├── tests/              # pytest 单元测试
├── main.py             # FastAPI 入口
└── requirements.txt
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

访问接口：

```bash
curl http://127.0.0.1:8000/api/evaluate/gold
```

## API

### `GET /api/evaluate/gold`

返回 JSON 格式黄金当前价位评估结果，包含：

- `symbol`：交易品种，例如 `XAUUSD`
- `price`：黄金现价，来自数据源
- `currency`：报价单位
- `data_status`：`ok` 或 `insufficient_data`
- `decision`：`增配`、`持有`、`观望`、`减配`、`回避` 或 `数据不足`
- `score`：宏观分、技术分、资金流分、总分和因子说明
- `market_data`：用于评分的原始 mock 数据
- `technical_indicators`：计算得到的 MA、MACD、RSI、ATR
- `summary`：中文摘要，所有数值来自结构化结果
- `risk_disclaimer`：风险提示

## 测试

```bash
pytest
```

## 后续计划

- 接入真实黄金行情数据源。
- 接入 DXY 数据源。
- 接入 FRED 宏观数据。
- 使用 SQLite 缓存数据快照与评估结果。
- 可选接入 OpenAI API：只允许基于结构化结果生成中文报告，不允许编造数值或仓位建议。
