# Aurum Matrix - 黄金当前价位评估工具

Aurum Matrix 是一个基于 **Python 3.11 + FastAPI** 的黄金当前价位评估工具。项目目标是评估当前黄金价格是否适合配置或交易，**不做确定性价格预测，不提供收益承诺**。

当前版本已接入真实数据源客户端：Alpha Vantage 用于 XAU/USD 与 DXY 日线数据，FRED 用于宏观利率/通胀预期数据。若 API Key、行情、宏观、资金流或持仓数据不足，接口会返回 `数据不足`，不会用 mock 数值替代真实输入。

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
├── data_sources/       # Alpha Vantage 行情、FRED 宏观数据、mock 测试数据与 SQLite 缓存
├── indicators/         # MA、MACD、RSI、ATR 等技术指标
├── scoring/            # 宏观分、技术分、资金流分、总分
├── decision_engine/    # 输出 增配/持有/观望/减配/回避/数据不足
├── llm/                # 可选中文报告生成；当前为非 LLM 占位实现
├── backend/services/   # 统一数据入口与完整黄金评估编排逻辑
├── services/           # 旧版兼容服务层
├── tests/              # pytest 单元测试
├── main.py             # FastAPI 入口
└── requirements.txt
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
export FRED_API_KEY=your_fred_key
# 可选：启用 FRED / Alpha Vantage 响应缓存
export CACHE_DB_PATH=.aurum_cache.db
uvicorn main:app --reload
```

访问接口：

```bash
curl http://127.0.0.1:8000/api/evaluate/gold
```

## 配置

运行时配置集中在 `backend/config.py`，支持以下环境变量：

- `DATA_MODE`：`real` 或 `mock`，默认 `real`
- `MARKET_PROVIDER`：当前支持 `alpha_vantage`
- `MACRO_PROVIDER`：当前支持 `fred`
- `FRED_API_KEY`：FRED API Key
- `ALPHA_VANTAGE_API_KEY`：Alpha Vantage API Key
- `CACHE_DB_PATH`：可选 SQLite 缓存路径
- `DEFAULT_DXY_SYMBOL`：DXY 数据供应商 symbol，默认 `DXY`

## API

### `GET /api/evaluate/gold`

支持可选 query 参数 `mode=mock` 或 `mode=real`，用于单次请求覆盖 `DATA_MODE` 配置。

返回 JSON 格式黄金当前价位评估结果，包含：

- `symbol`：交易品种，例如 `XAUUSD`
- `price`：黄金现价，来自数据源
- `currency`：报价单位
- `data_status`：`ok` 或 `insufficient_data`
- `market_state`：`强多`、`偏多`、`中性`、`偏空`、`强空` 或 `数据不足`
- `action`：`增配`、`持有`、`观望`、`减配`、`回避` 或 `数据不足`
- `indicators`：计算得到的 MA20、MA60、MA120、MACD、RSI、ATR
- `macro`：实际利率、美债收益率、通胀预期及 real_yield/DXY/ETF/CFTC 趋势字段
- `factor_input`：传入 `calculate_factor_score()` 的结构化输入；数据不足时为 `null`
- `score`：宏观分、技术分、资金流分、总分、市场状态、原因和风险
- `report_markdown`：基于结构化结果生成的中文 Markdown 报告
- `risk_disclaimer`：风险提示

## 测试

```bash
pytest
```

## 后续计划

- 接入 ETF 资金流、CFTC 持仓和地缘风险数据源。
- 接入真实 ETF 资金流和 CFTC 持仓后，在 `DATA_MODE=real` 下输出完整评分。
- 使用 SQLite 持久化完整评估结果。
- 可选接入 OpenAI API：只允许基于结构化结果生成中文报告，不允许编造数值或仓位建议。
