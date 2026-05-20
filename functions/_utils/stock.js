const DAY_SECONDS = 24 * 60 * 60;

export function jsonResponse(data, init = {}) {
  return new Response(JSON.stringify(data), {
    ...init,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...(init.headers || {}),
    },
  });
}

export function normalizeTicker(ticker) {
  const value = String(ticker || "").trim().replace(/^\$/, "").toUpperCase();
  if (!value) {
    throw new Error("銘柄コードを入力してください");
  }
  if (/^\d{4}$/.test(value)) {
    return `${value}.T`;
  }
  return value;
}

export function formatStockName(meta, ticker) {
  const label = meta?.longName || meta?.shortName || meta?.symbol || ticker;
  if (ticker.endsWith(".T")) {
    return `${label} (${ticker.replace(".T", "")})`;
  }
  return `${label} (${ticker})`;
}

export function calculateRsi(rows, period = 14) {
  const values = Array(rows.length).fill(null);
  const gains = Array(rows.length).fill(0);
  const losses = Array(rows.length).fill(0);

  for (let i = 1; i < rows.length; i += 1) {
    const delta = rows[i].close - rows[i - 1].close;
    gains[i] = delta > 0 ? delta : 0;
    losses[i] = delta < 0 ? -delta : 0;
  }

  for (let i = period - 1; i < rows.length; i += 1) {
    const gainSlice = gains.slice(i - period + 1, i + 1);
    const lossSlice = losses.slice(i - period + 1, i + 1);
    const avgGain = gainSlice.reduce((sum, value) => sum + value, 0) / period;
    const avgLoss = lossSlice.reduce((sum, value) => sum + value, 0) / period;
    const rs = avgGain / (avgLoss + 1e-10);
    values[i] = 100 - 100 / (1 + rs);
  }

  return values;
}

export function calculateBollingerBands(rows, period = 20, numStd = 3) {
  const lower = Array(rows.length).fill(null);
  const upper = Array(rows.length).fill(null);
  const lowerDeviation = Array(rows.length).fill(null);

  for (let i = period - 1; i < rows.length; i += 1) {
    const slice = rows.slice(i - period + 1, i + 1).map((row) => row.close);
    const average = slice.reduce((sum, value) => sum + value, 0) / period;
    const variance = slice.reduce((sum, value) => sum + (value - average) ** 2, 0) / (period - 1);
    const std = Math.sqrt(variance);
    upper[i] = average + std * numStd;
    lower[i] = average - std * numStd;
    lowerDeviation[i] = ((rows[i].close - lower[i]) / lower[i]) * 100;
  }

  return { upper, lower, lowerDeviation };
}

export function enrichStockRows(rows) {
  const rsi = calculateRsi(rows);
  const { lowerDeviation } = calculateBollingerBands(rows);

  return rows.map((row, index) => {
    const rsiValue = Number.isFinite(rsi[index]) ? round(rsi[index], 2) : null;
    const lowerDeviationValue = Number.isFinite(lowerDeviation[index])
      ? round(lowerDeviation[index], 2)
      : null;
    let shortTermExpectation = null;

    if (rsiValue !== null && lowerDeviationValue !== null) {
      const rsiComponent = 50 - rsiValue;
      const deviationComponent = Math.abs(Math.min(0, lowerDeviationValue));
      const rawExpectation = rsiComponent + deviationComponent;
      shortTermExpectation = rawExpectation >= 0
        ? round(rawExpectation * 2, 2)
        : round(rawExpectation, 2);
    }

    return {
      ...row,
      rsi: rsiValue,
      lower_deviation: lowerDeviationValue,
      short_term_expectation: shortTermExpectation,
    };
  });
}

export async function fetchStockData(tickerInput, periodDays = 730) {
  const ticker = normalizeTicker(tickerInput);
  const now = Math.floor(Date.now() / 1000);
  const period = clampInteger(periodDays, 30, 3650, 730);
  const period1 = now - period * DAY_SECONDS;

  const payload = await fetchYahooChart(ticker, period, period1, now);
  const result = payload?.chart?.result?.[0];
  const error = payload?.chart?.error;

  if (error) {
    throw new Error(error.description || "株価データの取得に失敗しました");
  }
  if (!result?.timestamp?.length) {
    throw new Error(`銘柄コード '${tickerInput}' のデータが見つかりませんでした。`);
  }

  const quote = result.indicators?.quote?.[0] || {};
  const rows = result.timestamp.map((timestamp, index) => {
    const open = quote.open?.[index];
    const high = quote.high?.[index];
    const low = quote.low?.[index];
    const close = quote.close?.[index];
    const volume = quote.volume?.[index];

    if (![open, high, low, close].every(Number.isFinite)) {
      return null;
    }

    return {
      date: new Date(timestamp * 1000).toISOString().slice(0, 10),
      open,
      high,
      low,
      close,
      volume: Number.isFinite(volume) ? volume : 0,
    };
  }).filter(Boolean);

  if (rows.length < 20) {
    throw new Error(`銘柄コード '${tickerInput}' の分析に必要なデータが不足しています。`);
  }

  return {
    ticker,
    stock_name: formatStockName(result.meta, ticker),
    stock_data: enrichStockRows(rows).sort((a, b) => b.date.localeCompare(a.date)),
  };
}

async function fetchYahooChart(ticker, period, period1, period2) {
  const hosts = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"];
  const errors = [];

  for (const host of hosts) {
    for (const mode of ["period", "range"]) {
      const url = new URL(`https://${host}/v8/finance/chart/${encodeURIComponent(ticker)}`);
      url.searchParams.set("interval", "1d");
      url.searchParams.set("events", "history");
      url.searchParams.set("includeAdjustedClose", "true");

      if (mode === "period") {
        url.searchParams.set("period1", String(period1));
        url.searchParams.set("period2", String(period2));
      } else {
        url.searchParams.set("range", `${period}d`);
      }

      try {
        const response = await fetch(url.toString(), {
          headers: {
            "user-agent": "Mozilla/5.0 stock-expectation-cloudflare",
            "accept": "application/json",
          },
        });

        if (!response.ok) {
          errors.push(`HTTP ${response.status}`);
          continue;
        }

        return await response.json();
      } catch (error) {
        errors.push(error.message);
      }
    }
  }

  throw new Error(`株価データの取得に失敗しました: ${errors.join(", ")}`);
}

export function calculateBacktest(stockData, options = {}) {
  const buyThreshold = Number(options.buy_threshold ?? 30);
  const disableSell = Boolean(options.disable_sell);
  const sellThreshold = disableSell ? null : Number(options.sell_threshold ?? 0);
  const shares = clampInteger(options.shares, 1, 10_000_000, 100);
  const trades = [];
  let positions = [];

  const sortedDesc = [...stockData].sort((a, b) => b.date.localeCompare(a.date));
  const latestData = sortedDesc[0];
  const latestPrice = latestData.close;
  const latestDate = latestData.date;
  const sortedAsc = [...stockData].sort((a, b) => a.date.localeCompare(b.date));

  for (const currentData of sortedAsc) {
    const currentTime = Date.parse(currentData.date);
    const expectation = currentData.short_term_expectation;

    if (expectation !== null && expectation >= buyThreshold) {
      const position = {
        buy_date: currentData.date,
        buy_price: currentData.close,
        shares,
        buy_expectation: expectation,
        buy_time: currentTime,
      };
      positions.push(position);

      if (disableSell) {
        trades.push(buildTrade(position, latestDate, latestPrice, null));
      }
    } else if (
      !disableSell
      && sellThreshold !== null
      && expectation !== null
      && expectation <= sellThreshold
      && positions.length > 0
    ) {
      for (const position of positions) {
        if (currentTime > position.buy_time) {
          trades.push(buildTrade(position, currentData.date, currentData.close, expectation));
        }
      }
      positions = [];
    }
  }

  if (positions.length > 0 && !disableSell) {
    const latestTime = Date.parse(latestDate);
    for (const position of positions) {
      if (latestTime > position.buy_time) {
        trades.push(buildTrade(position, latestDate, latestPrice, latestData.short_term_expectation));
      }
    }
  }

  return {
    trades,
    performance: summarizeTrades(trades, disableSell),
  };
}

function buildTrade(position, sellDate, sellPrice, sellExpectation) {
  const profitRate = ((sellPrice - position.buy_price) / position.buy_price) * 100;
  return {
    buy_date: position.buy_date,
    sell_date: sellDate,
    buy_price: position.buy_price,
    sell_price: sellPrice,
    shares: position.shares,
    buy_expectation: position.buy_expectation,
    sell_expectation: sellExpectation,
    profit_rate: profitRate,
    profit_amount: (sellPrice - position.buy_price) * position.shares,
  };
}

function summarizeTrades(trades, disableSell) {
  const totalTrades = trades.length;
  if (totalTrades === 0) {
    return {
      total_trades: 0,
      winning_trades: 0,
      losing_trades: 0,
      win_rate: 0,
      total_profit_rate: 0,
      total_profit_amount: 0,
      average_profit_rate: 0,
      max_profit_rate: 0,
    };
  }

  const winningTrades = trades.filter((trade) => trade.profit_rate > 0).length;
  const sellDateProfits = {};
  for (const trade of trades) {
    sellDateProfits[trade.sell_date] ||= [];
    sellDateProfits[trade.sell_date].push(trade.profit_rate);
  }

  const avgDailyProfits = Object.values(sellDateProfits)
    .map((profits) => profits.reduce((sum, value) => sum + value, 0) / profits.length);
  let totalProfitRate = (avgDailyProfits.reduce((acc, rate) => acc * (1 + rate / 100), 1) - 1) * 100;
  const averageProfitRate = trades.reduce((sum, trade) => sum + trade.profit_rate, 0) / totalTrades;

  if (disableSell) {
    totalProfitRate = averageProfitRate;
  }

  return {
    total_trades: totalTrades,
    winning_trades: winningTrades,
    losing_trades: totalTrades - winningTrades,
    win_rate: (winningTrades / totalTrades) * 100,
    total_profit_rate: totalProfitRate,
    total_profit_amount: trades.reduce((sum, trade) => sum + trade.profit_amount, 0),
    average_profit_rate: averageProfitRate,
    max_profit_rate: Math.max(...trades.map((trade) => trade.profit_rate)),
  };
}

function clampInteger(value, min, max, fallback) {
  const number = Number.parseInt(value, 10);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, number));
}

function round(value, digits) {
  const factor = 10 ** digits;
  return Math.round((value + Number.EPSILON) * factor) / factor;
}
