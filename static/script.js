document.addEventListener('DOMContentLoaded', () => {
    const tickerInput = document.getElementById('tickerInput');
    const fetchButton = document.getElementById('fetchButton');
    const stockListTable = document.getElementById('stockListTable');
    const stockListBody = document.getElementById('stockListBody');
    const tickerTitle = document.getElementById('tickerTitle');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorDisplay = document.getElementById('errorDisplay');

    fetchButton.addEventListener('click', fetchData);
    tickerInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            fetchData();
        }
    });

    async function fetchData() {
        const ticker = tickerInput.value.trim();
        if (!ticker) {
            showError('銘柄コードを入力してください。');
            return;
        }

        // UIリセット
        clearResults();
        showLoading(true);

        try {
            const response = await fetch('/get_stock_list', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `ticker=${encodeURIComponent(ticker)}`
            });

            const result = await response.json();

            showLoading(false);

            if (!response.ok || result.error) {
                showError(result.error || `エラーが発生しました (ステータス: ${response.status})`);
            } else if (result.stock_data && result.stock_data.length > 0) {
                displayStockList(ticker, result.stock_data);
            } else {
                showError('データが見つかりませんでした。');
            }

        } catch (error) {
            console.error('Fetch error:', error);
            showLoading(false);
            showError('データの取得中にエラーが発生しました。');
        }
    }

    function displayStockList(ticker, data) {
        tickerTitle.textContent = `銘柄コード: ${ticker}`; // 銘柄コードを表示
        stockListBody.innerHTML = ''; // テーブル内容をクリア

        data.forEach(item => {
            const row = stockListBody.insertRow();
            row.insertCell().textContent = item.date;
            row.insertCell().textContent = item.open.toLocaleString(); // 見やすくするためにカンマ区切り
            row.insertCell().textContent = item.high.toLocaleString();
            row.insertCell().textContent = item.low.toLocaleString();
            row.insertCell().textContent = item.close.toLocaleString();
            row.insertCell().textContent = item.volume.toLocaleString();
        });

        stockListTable.style.display = 'table'; // テーブルを表示
    }

    function showLoading(isLoading) {
        loadingIndicator.style.display = isLoading ? 'block' : 'none';
    }

    function showError(message) {
        errorDisplay.textContent = message;
        errorDisplay.style.display = 'block';
    }

    function clearResults() {
        stockListBody.innerHTML = '';
        tickerTitle.textContent = '';
        errorDisplay.textContent = '';
        errorDisplay.style.display = 'none';
        stockListTable.style.display = 'none'; // 最初はテーブルを隠す
    }

    function displayStockData(data) {
        const tableBody = document.getElementById('stockDataBody');
        tableBody.innerHTML = '';

        data.forEach(item => {
            const row = document.createElement('tr');
            
            // 数値の変換を確実に行う
            const rsi = parseFloat(item.rsi) || 0;
            const lowerDeviation = parseFloat(item.lower_deviation) || 0;
            const shortTermExpectation = parseFloat(item.short_term_expectation) || 0;
            
            row.innerHTML = `
                <td>${item.date}</td>
                <td>${formatNumber(item.open)}</td>
                <td>${formatNumber(item.high)}</td>
                <td>${formatNumber(item.low)}</td>
                <td>${formatNumber(item.close)}</td>
                <td>${formatNumber(item.volume)}</td>
                <td class="${getRSIClass(rsi)}">${formatNumber(rsi)}</td>
                <td class="${getDeviationClass(lowerDeviation)}">${formatNumber(lowerDeviation)}</td>
                <td class="${getExpectationClass(shortTermExpectation)}">${formatNumber(shortTermExpectation)}</td>
            `;
            tableBody.appendChild(row);
        });
    }
}); 