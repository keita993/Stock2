document.getElementById('ticker-form').addEventListener('submit', function(event) {
    event.preventDefault(); // デフォルトのフォーム送信をキャンセル

    const ticker = document.getElementById('ticker').value;
    const selectedModel = document.getElementById('model-select').value; // 選択されたモデルの値を取得
    const resultArea = document.getElementById('result-area');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error-message');
    const historicalTbody = document.getElementById('historical-tbody');

    // 結果表示とエラーメッセージをクリア、ローディング表示
    resultArea.style.display = 'none';
    errorDiv.style.display = 'none';
    errorDiv.textContent = '';
    historicalTbody.innerHTML = ''; // テーブルの内容をクリア
    loadingDiv.style.display = 'block';

    // FormDataオブジェクトを作成してtickerとmodelをセット
    const formData = new FormData();
    formData.append('ticker', ticker);
    formData.append('model', selectedModel); // モデルの値を追加

    // バックエンドAPIにPOSTリクエストを送信
    fetch('/get_data', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            // エラーレスポンスの場合はエラーメッセージを取得して表示
            return response.json().then(err => {
                throw new Error(err.error || `HTTP error! status: ${response.status}`);
            });
        }
        return response.json(); // 正常なレスポンスの場合はJSONをパース
    })
    .then(data => {
        // --- 最新データの表示 ---
        document.getElementById('info-ticker').textContent = data.latest.ticker;
        document.getElementById('info-price').textContent = data.latest.current_price;
        document.getElementById('latest-index-value').textContent = data.latest.expectation_index;
        document.getElementById('latest-index-status').textContent = data.latest.status;
        document.getElementById('latest-index-status').className = getStatusClass(data.latest.status); // 状態クラスを適用
        document.getElementById('latest-volume').textContent = data.latest.volume.toLocaleString(); // 桁区切り表示
        document.getElementById('latest-rsi').textContent = data.latest.rsi;
        document.getElementById('latest-bb-lower').textContent = data.latest.bollinger_bands.lower;
        document.getElementById('latest-bb-middle').textContent = data.latest.bollinger_bands.middle;
        document.getElementById('latest-bb-upper').textContent = data.latest.bollinger_bands.upper;

        // --- 時系列データの表示 ---
        if (data.historical_data && data.historical_data.length > 0) {
            data.historical_data.forEach(item => {
                const row = historicalTbody.insertRow();
                row.insertCell().textContent = item.date;
                row.insertCell().textContent = item.open;
                row.insertCell().textContent = item.high;
                row.insertCell().textContent = item.low;
                row.insertCell().textContent = item.close;
                row.insertCell().textContent = item.volume.toLocaleString(); // 桁区切り表示
                row.insertCell().textContent = item.expectation_index;
                const statusCell = row.insertCell();
                statusCell.textContent = item.status;
                statusCell.className = getStatusClass(item.status); // 状態クラスを適用
            });
        }

        // ローディングを非表示、結果を表示
        loadingDiv.style.display = 'none';
        resultArea.style.display = 'block';
    })
    .catch(error => {
        // エラーが発生した場合
        console.error('Error:', error);
        errorDiv.textContent = `エラーが発生しました: ${error.message}`;
        errorDiv.style.display = 'block';
        loadingDiv.style.display = 'none';
        resultArea.style.display = 'none'; // エラー時は結果を非表示
    });
});

// 状態テキストに基づいてCSSクラスを返すヘルパー関数
function getStatusClass(statusText) {
    if (!statusText) return '';
    const lowerStatus = statusText.toLowerCase();
    if (lowerStatus.includes('買い') || lowerStatus.includes('buy')) {
        return 'status-buy';
    } else if (lowerStatus.includes('売り') || lowerStatus.includes('sell')) {
        return 'status-sell';
    } else {
        return 'status-neutral'; // デフォルトまたは中立の場合
    }
}

// 初期状態では結果を非表示にする
document.addEventListener('DOMContentLoaded', (event) => {
    const resultArea = document.getElementById('result-area');
    if (resultArea) {
        resultArea.style.display = 'none';
    }
});

// 新規登録フォームの処理
document.getElementById('registerFormElement').addEventListener('submit', async function(e) {
    e.preventDefault();
    const username = document.getElementById('newUsername').value;
    const password = document.getElementById('newPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;

    if (password !== confirmPassword) {
        alert('パスワードが一致しません');
        return;
    }

    try {
        const response = await fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.message || '登録に失敗しました');
        }

        if (data.success) {
            alert('登録が完了しました。ログインしてください。');
            document.getElementById('showLoginForm').click();
        } else {
            alert('登録に失敗しました: ' + data.message);
        }
    } catch (error) {
        console.error('Registration error:', error);
        alert('登録中にエラーが発生しました: ' + error.message);
    }
}); 