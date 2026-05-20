import { calculateBacktest, fetchStockData, jsonResponse } from "../_utils/stock.js";

export async function onRequestPost({ request }) {
  try {
    const body = await request.json();
    const result = await fetchStockData(body.ticker, body.period);
    const backtest = calculateBacktest(result.stock_data, body);
    return jsonResponse({
      ...backtest,
      stock_data: result.stock_data,
      stock_name: result.stock_name,
      ticker: result.ticker,
    });
  } catch (error) {
    return jsonResponse({ error: error.message || "バックテスト中にエラーが発生しました" }, { status: 400 });
  }
}

export async function onRequestGet() {
  return jsonResponse({ error: "POSTでバックテスト条件を指定してください" }, { status: 405 });
}
