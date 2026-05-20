import { fetchStockData, jsonResponse } from "../_utils/stock.js";

export async function onRequestPost({ request }) {
  try {
    const body = await request.json();
    const result = await fetchStockData(body.ticker, body.period);
    return jsonResponse(result);
  } catch (error) {
    return jsonResponse({ error: error.message || "データ取得中にエラーが発生しました" }, { status: 400 });
  }
}

export async function onRequestGet() {
  return jsonResponse({ error: "POSTで銘柄コードを指定してください" }, { status: 405 });
}
