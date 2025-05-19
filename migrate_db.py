from app import app, db, Trade
from datetime import datetime

def migrate_database():
    with app.app_context():
        # 既存の取引データを取得
        trades = Trade.query.all()
        
        # 各取引の株数を100株に設定
        for trade in trades:
            if not hasattr(trade, 'quantity'):
                trade.quantity = 100
        
        # 変更を保存
        db.session.commit()
        print("データベースのマイグレーションが完了しました")

if __name__ == "__main__":
    migrate_database() 