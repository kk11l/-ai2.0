import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AI Quant Trader A股量化平台",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Quant Trader")
st.caption("免费 A 股量化研究平台 | 技术分析 | 策略回测")

# -----------------------
# 侧边栏参数
# -----------------------

st.sidebar.header("策略参数")

symbol = st.sidebar.text_input(
    "A股代码",
    value="600519"
).strip()

short_window = st.sidebar.slider(
    "短期均线",
    min_value=3,
    max_value=30,
    value=10
)

long_window = st.sidebar.slider(
    "长期均线",
    min_value=20,
    max_value=120,
    value=60
)

initial_cash = st.sidebar.number_input(
    "初始模拟资金（元）",
    min_value=1000,
    max_value=100000000,
    value=100000,
    step=10000
)

fee_rate = st.sidebar.number_input(
    "单边交易成本（估算）",
    min_value=0.0,
    max_value=1.0,
    value=0.0005,
    step=0.0001,
    format="%.4f"
)

st.sidebar.caption(
    "交易成本仅为估算值，请根据实际佣金、税费和滑点调整。"
)

if short_window >= long_window:
    st.error("短期均线必须小于长期均线。")
    st.stop()

# -----------------------
# 获取历史行情
# -----------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_history(code):
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (
        datetime.now() - timedelta(days=365 * 5)
    ).strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )

    if df.empty:
        raise ValueError("行情数据为空")

    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount"
    })

    df["date"] = pd.to_datetime(df["date"])

    for col in [
        "open", "close", "high",
        "low", "volume", "amount"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date")
    df = df.drop_duplicates("date")

    return df.reset_index(drop=True)


# -----------------------
# 技术指标
# -----------------------

def calculate_indicators(df):
    df = df.copy()

    df["MA_SHORT"] = (
        df["close"].rolling(short_window).mean()
    )

    df["MA_LONG"] = (
        df["close"].rolling(long_window).mean()
    )

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - 100 / (1 + rs)

    ema12 = df["close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    return df


# -----------------------
# 策略回测
# -----------------------

def backtest(df):
    df = df.copy()

    # 均线金叉/死叉策略
    df["position"] = (
        df["MA_SHORT"] > df["MA_LONG"]
    ).astype(int)

    # 使用上一交易日收盘后形成的仓位，
    # 避免直接使用当日收盘信号获得当日收益
    df["held_position"] = df["position"].shift(1).fillna(0)

    df["market_return"] = (
        df["close"].pct_change().fillna(0)
    )

    # 仅在仓位变化时估算交易成本
    df["turnover"] = (
        df["position"].diff().abs().fillna(
            df["position"].abs()
        )
    )

    df["strategy_return"] = (
        df["held_position"] * df["market_return"]
        - df["turnover"] * fee_rate
    )

    df["strategy_equity"] = (
        initial_cash
        * (1 + df["strategy_return"]).cumprod()
    )

    df["buy_hold_equity"] = (
        initial_cash
        * (1 + df["market_return"]).cumprod()
    )

    df["drawdown"] = (
        df["strategy_equity"]
        / df["strategy_equity"].cummax() - 1
    )

    return df


# -----------------------
# 主程序
# -----------------------

if st.button("🔄 刷新行情"):
    get_history.clear()

try:
    with st.spinner("正在获取行情并计算指标..."):
        raw = get_history(symbol)

        data = calculate_indicators(raw)

        data = backtest(data)

    latest = data.iloc[-1]

    st.caption(
        f"股票代码：{symbol} | "
        f"最新数据日期：{latest['date'].date()} | "
        f"行情更新时间以数据源为准"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "最新收盘价",
        f"¥{latest['close']:.2f}"
    )

    if len(data) > 1:
        change = (
            latest["close"] / data.iloc[-2]["close"] - 1
        ) * 100
    else:
        change = 0

    col2.metric(
        "较上一交易日",
        f"{change:.2f}%"
    )

    col3.metric(
        "RSI",
        f"{latest['RSI']:.2f}"
        if pd.notna(latest["RSI"])
        else "计算中"
    )

    col4.metric(
        "模拟资金",
        f"¥{latest['strategy_equity']:,.0f}"
    )

    # 信号分析
    st.subheader("🤖 量化信号分析")

    if (
        latest["MA_SHORT"] > latest["MA_LONG"]
        and latest["MACD"] > latest["SIGNAL"]
        and latest["RSI"] < 70
    ):
        st.success(
            "规则信号：趋势偏强。"
            "这不是保证盈利的买入建议。"
        )

    elif (
        latest["MA_SHORT"] < latest["MA_LONG"]
        and latest["MACD"] < latest["SIGNAL"]
    ):
        st.error(
            "规则信号：趋势偏弱，注意回撤风险。"
        )

    else:
        st.warning(
            "规则信号：趋势不一致，建议继续观察。"
        )

    st.caption(
        "当前版本采用技术指标规则，不是已经训练的机器学习模型。"
    )

    # 价格图
    st.subheader("📊 收盘价与均线")

    price_chart = data.set_index("date")[
        ["close", "MA_SHORT", "MA_LONG"]
    ]

    st.line_chart(price_chart)

    # MACD
    st.subheader("MACD")

    st.line_chart(
        data.set_index("date")[["MACD", "SIGNAL"]]
    )

    # 回测指标
    st.subheader("🧪 历史策略回测")

    total_return = (
        latest["strategy_equity"] / initial_cash - 1
    ) * 100

    max_drawdown = (
        data["drawdown"].min() * 100
    )

    daily_returns = data["strategy_return"]

    if daily_returns.std() > 0:
        sharpe = (
            daily_returns.mean()
            / daily_returns.std()
            * np.sqrt(252)
        )
    else:
        sharpe = 0

    years = max(
        (data["date"].iloc[-1] - data["date"].iloc[0]).days
        / 365.25,
        0.01
    )

    cagr = (
        (latest["strategy_equity"] / initial_cash)
        ** (1 / years) - 1
    ) * 100

    a, b, c, d = st.columns(4)

    a.metric("累计收益", f"{total_return:.2f}%")
    b.metric("年化收益估算", f"{cagr:.2f}%")
    c.metric("最大回撤", f"{max_drawdown:.2f}%")
    d.metric("夏普比率估算", f"{sharpe:.2f}")

    st.line_chart(
        data.set_index("date")[
            ["strategy_equity", "buy_hold_equity"]
        ]
    )

    st.caption(
        "回测未完全模拟A股涨跌停、停牌、成交限制和全部交易摩擦。"
        "结果不代表未来收益。"
    )

    # 导出数据
    st.subheader("📥 导出分析结果")

    csv_data = data.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "下载回测 CSV",
        data=csv_data,
        file_name=f"{symbol}_backtest.csv",
        mime="text/csv"
    )

    st.subheader("最新行情数据")

    st.dataframe(
        data.tail(20),
        use_container_width=True
    )

except Exception as e:
    st.error(
        "获取行情或计算失败。"
        "请检查股票代码、网络连接及数据源是否可用。"
    )

    st.caption(f"错误详情：{e}")
