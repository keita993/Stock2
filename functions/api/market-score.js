import { fetchStockData, jsonResponse } from "../_utils/stock.js";

const MARKETS = [
  { key: "nikkei", label: "日経平均", ticker: "^N225", source: "Yahoo Finance" },
  { key: "growth", label: "グロース250", ticker: "2516", source: "東証グロース250 ETF proxy" },
];

const MACRO_SERIES = {
  vix: { label: "VIX", ticker: "^VIX", source: "Yahoo Finance" },
  nikkeiVi: { label: "日経VI", tickers: ["^NKVF.OS", "^N225VI"], source: "Yahoo Finance proxy" },
  nasdaq: { label: "NASDAQ", ticker: "^IXIC", source: "Yahoo Finance" },
  sox: { label: "SOX", ticker: "^SOX", source: "Yahoo Finance" },
};

export async function onRequestGet() {
  try {
    return jsonResponse(await buildMarketScore());
  } catch (error) {
    return jsonResponse({ error: error.message || "市場スコアの取得に失敗しました" }, { status: 400 });
  }
}

export async function onRequestPost() {
  return onRequestGet();
}

async function buildMarketScore() {
  const [marketResults, vix, nikkeiVi, nasdaq, sox] = await Promise.all([
    Promise.all(MARKETS.map((market) => loadSeries(market.ticker, market))),
    loadSeries(MACRO_SERIES.vix.ticker, MACRO_SERIES.vix),
    loadFirstAvailable(MACRO_SERIES.nikkeiVi.tickers, MACRO_SERIES.nikkeiVi),
    loadSeries(MACRO_SERIES.nasdaq.ticker, MACRO_SERIES.nasdaq),
    loadSeries(MACRO_SERIES.sox.ticker, MACRO_SERIES.sox),
  ]);

  const sharedComponents = {
    vix: buildLatestThresholdComponent("vix", "VIX 30以上", vix, 30),
    nikkeiVi: buildLatestThresholdComponent("nikkeiVi", "日経VI 25以上", nikkeiVi, 25),
    nasdaq: buildReturnThresholdComponent("nasdaq", "NASDAQ 4週 -5%以下", nasdaq, -5),
    sox: buildReturnThresholdComponent("sox", "SOX 4週 -8%以下", sox, -8),
    arbitrage: {
      key: "arbitrage",
      label: "裁定買い残 5000億円以下",
      active: false,
      passed: false,
      points: 0,
      value: null,
      formatted: "未接続",
      source: "JPX Excel解析を後続実装",
    },
  };

  const markets = marketResults.map((result) => {
    if (!result.ok) {
      return {
        key: result.meta.key,
        label: result.meta.label,
        ticker: result.meta.ticker,
        ok: false,
        error: result.error,
        score: 0,
        maxScore: 0,
        components: [],
      };
    }

    const deviation = calculateMaDeviation(result.rows, 25);
    const components = [
      {
        key: "maDeviation25",
        label: "25日線乖離率 -5%以下",
        active: deviation !== null,
        passed: deviation !== null && deviation <= -5,
        points: deviation !== null && deviation <= -5 ? 1 : 0,
        value: deviation,
        formatted: deviation === null ? "-" : `${formatNumber(deviation)}%`,
        source: result.meta.source,
      },
      sharedComponents.vix,
      sharedComponents.nikkeiVi,
      sharedComponents.nasdaq,
      sharedComponents.sox,
      sharedComponents.arbitrage,
    ];
    const activeComponents = components.filter((component) => component.active);
    const score = activeComponents.reduce((sum, component) => sum + component.points, 0);

    return {
      key: result.meta.key,
      label: result.meta.label,
      ticker: result.ticker,
      ok: true,
      latestDate: result.rows[0]?.date || null,
      latestClose: result.rows[0]?.close ?? null,
      score,
      maxScore: activeComponents.length,
      status: score >= 4 ? "強い買いシグナル" : score >= 3 ? "買い候補" : score >= 2 ? "監視" : "通常",
      components,
    };
  });

  return {
    generatedAt: new Date().toISOString(),
    version: "lite-without-credit-evaluation",
    title: "BUY SCORE Lite",
    description: "信用評価損益率を除外し、取得できる市場データのみで採点します。",
    markets,
    shared: {
      vix: summarizeSeries(vix),
      nikkeiVi: summarizeSeries(nikkeiVi),
      nasdaq: summarizeSeries(nasdaq, true),
      sox: summarizeSeries(sox, true),
    },
  };
}

async function loadFirstAvailable(tickers, meta) {
  let lastError = null;
  for (const ticker of tickers) {
    const result = await loadSeries(ticker, { ...meta, ticker });
    if (result.ok) return result;
    lastError = result.error;
  }
  return { ok: false, meta, rows: [], error: lastError || "データを取得できませんでした" };
}

async function loadSeries(ticker, meta) {
  try {
    const data = await fetchStockData(ticker, 180);
    return {
      ok: true,
      meta: { ...meta, ticker },
      ticker: data.ticker,
      rows: data.stock_data,
    };
  } catch (error) {
    return {
      ok: false,
      meta: { ...meta, ticker },
      rows: [],
      error: error.message || "データを取得できませんでした",
    };
  }
}

function buildLatestThresholdComponent(key, label, result, threshold) {
  const latest = result.ok ? result.rows[0]?.close : null;
  return {
    key,
    label,
    active: Number.isFinite(latest),
    passed: Number.isFinite(latest) && latest >= threshold,
    points: Number.isFinite(latest) && latest >= threshold ? 1 : 0,
    value: Number.isFinite(latest) ? round(latest, 2) : null,
    formatted: Number.isFinite(latest) ? formatNumber(latest) : "取得不可",
    source: result.meta?.source || "-",
    error: result.ok ? null : result.error,
  };
}

function buildReturnThresholdComponent(key, label, result, threshold) {
  const return20 = result.ok ? calculateTradingDayReturn(result.rows, 20) : null;
  return {
    key,
    label,
    active: return20 !== null,
    passed: return20 !== null && return20 <= threshold,
    points: return20 !== null && return20 <= threshold ? 1 : 0,
    value: return20,
    formatted: return20 === null ? "取得不可" : `${formatNumber(return20)}%`,
    source: result.meta?.source || "-",
    error: result.ok ? null : result.error,
  };
}

function calculateMaDeviation(rows, period) {
  if (!Array.isArray(rows) || rows.length < period) return null;
  const slice = rows.slice(0, period).map((row) => row.close).filter(Number.isFinite);
  if (slice.length < period) return null;
  const latest = rows[0].close;
  const average = slice.reduce((sum, value) => sum + value, 0) / period;
  return round(((latest - average) / average) * 100, 2);
}

function calculateTradingDayReturn(rows, days) {
  if (!Array.isArray(rows) || rows.length <= days) return null;
  const latest = rows[0].close;
  const previous = rows[days].close;
  if (!Number.isFinite(latest) || !Number.isFinite(previous) || previous === 0) return null;
  return round(((latest - previous) / previous) * 100, 2);
}

function summarizeSeries(result, includeReturn = false) {
  if (!result.ok) {
    return {
      ok: false,
      label: result.meta?.label || "-",
      source: result.meta?.source || "-",
      error: result.error,
    };
  }
  return {
    ok: true,
    label: result.meta.label,
    ticker: result.ticker,
    source: result.meta.source,
    latestDate: result.rows[0]?.date || null,
    latestClose: result.rows[0]?.close ?? null,
    return20: includeReturn ? calculateTradingDayReturn(result.rows, 20) : null,
  };
}

function formatNumber(value) {
  return new Intl.NumberFormat("ja-JP", {
    maximumFractionDigits: 2,
    minimumFractionDigits: 0,
  }).format(value);
}

function round(value, digits) {
  const factor = 10 ** digits;
  return Math.round((value + Number.EPSILON) * factor) / factor;
}
