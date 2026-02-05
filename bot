API_KEY = "zCZGYecjkpzaYz1zZqrQBBDR5BWogKo7ly8YV4nuptuRRZke8jbn8rhU0e8FXYL"
API_SECRET = "A8pDvacuGVR6TOrkFfioWt9vNdg94296cgZXlbNOINw1jJBGWTQRknWOOF5PBvPq"
TELEGRAM_TOKEN = "5387224565:AAFjj8TpdxgBSyvFGoLNLV-NsZg8z0RxetQ"
TELEGRAM_CHAT_ID = "1733841300"
# ====================== CONFIG & IMPORTS ======================
import time, math, datetime, pickle, os, requests
import pandas as pd
import pandas_ta as ta
from threading import Thread
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.ensemble import RandomForestClassifier
from binance.um_futures import UMFutures

# CẤU HÌNH TÀI KHOẢN
SYMBOL = "ADAUSDT"
LEVERAGE = 25
BASE_URL = "https://testnet.binancefuture.com" 

client = UMFutures(API_KEY, API_SECRET, base_url=BASE_URL)
MODEL_PATH = "./models"
os.makedirs(MODEL_PATH, exist_ok=True)

# BIẾN TOÀN CỤC
last_order_candle = None
monitoring_sides = set()
BEST_PARAMS = {"tp_roe": 100.0, "sl_roe": 50.0} 
FEATURES = ["RSI", "EMA_fast", "EMA_slow", "ATR", "ROC", "body_size", "upper_shadow"]

# ====================== UTILS ======================
def sync_time():
    try:
        server_time = client.time()["serverTime"]
        client.timestamp_offset = server_time - int(time.time() * 1000)
    except: pass

def send_tele(msg):
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                     data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)
    except: pass

def get_symbol_filters():
    try:
        info = client.exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == SYMBOL:
                lot = next(f for f in s["filters"] if f["filterType"] == "LOT_SIZE")
                qty_p = int(round(-math.log(float(lot["stepSize"]), 10)))
                price_f = next(f for f in s["filters"] if f["filterType"] == "PRICE_FILTER")
                price_p = int(round(-math.log(float(price_f["tickSize"]), 10)))
                return float(lot["minQty"]), qty_p, price_p
    except: pass
    return 0.1, 1, 4

def get_open_positions():
    try:
        pos = client.get_position_risk(symbol=SYMBOL)
        return [{"side": p["positionSide"], "qty": abs(float(p["positionAmt"])), "entry": float(p["entryPrice"])} 
                for p in pos if float(p["positionAmt"]) != 0]
    except: return []

def calc_qty(price):
    min_q, qty_p, _ = get_symbol_filters()
    return round(max(5.0 / price, min_q), qty_p)

# ====================== DATA & AI ======================
def prepare_features(df):
    column_names = ["t","open","high","low","close","vol","ct","q","nt","tb","tq","i"]
    df.columns = column_names[:len(df.columns)] 
    for c in ["open","high","low","close","vol"]: df[c] = df[c].astype(float)
    
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["EMA_fast"] = ta.ema(df["close"], length=9)
    df["EMA_slow"] = ta.ema(df["close"], length=21)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ROC"] = ta.roc(df["close"], length=5)
    df["body_size"] = abs(df["close"] - df["open"])
    df["upper_shadow"] = df["high"] - df.loc[:, ["open", "close"]].max(axis=1)
    return df.dropna()

def optimize_trading_params():
    global BEST_PARAMS
    try:
        kl = client.klines(SYMBOL, "1m", limit=1000)
        df_bt = prepare_features(pd.DataFrame(kl))
        def obj(t):
            tp = t.suggest_float("tp", 30, 150)
            sl = t.suggest_float("sl", 20, 80)
            balance = 100.0
            for i in range(1, len(df_bt)):
                change = (df_bt['close'].iloc[i] - df_bt['close'].iloc[i-1])/df_bt['close'].iloc[i-1]
                roe = change * LEVERAGE * 100
                if roe >= tp: balance += balance * (tp/100)
                elif roe <= -sl: balance -= balance * (sl/100)
            return balance
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=30)
        BEST_PARAMS = {"tp_roe": round(study.best_params["tp"], 2), "sl_roe": round(study.best_params["sl"], 2)}
    except: BEST_PARAMS = {"tp_roe": 100.0, "sl_roe": 50.0}

def train_models():
    send_tele("🧠 **AI ĐANG HỌC LẠI DỮ LIỆU & TỐI ƯU CHIẾN THUẬT...**")
    kl = client.klines(SYMBOL, "15m", limit=1500)
    df = prepare_features(pd.DataFrame(kl))
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    X, y = df[FEATURES], df["target"]
    
    m1 = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, verbosity=0)
    m2 = RandomForestClassifier(n_estimators=100, max_depth=6)
    m3 = lgb.LGBMClassifier(n_estimators=100, verbosity=-1)
    for m in [m1, m2, m3]: m.fit(X, y)
    
    pickle.dump([m1, m2, m3], open(f"{MODEL_PATH}/models.pkl", "wb"))
    optimize_trading_params()
    send_tele(f"✅ **DONE!** AI chốt mục tiêu mới: TP `{BEST_PARAMS['tp_roe']}%` | SL `{BEST_PARAMS['sl_roe']}%`")

# ====================== EXECUTION & MONITOR ======================
def place_tp_sl(entry, side, custom_sl_roe=None):
    try:
        _, _, price_p = get_symbol_filters()
        # Nếu không có custom_sl_roe thì lấy SL âm mặc định của AI
        sl_roe = custom_sl_roe if custom_sl_roe is not None else -BEST_PARAMS['sl_roe']
        sl_price = round(entry * (1 + sl_roe/(100*LEVERAGE)), price_p) if side=="LONG" else round(entry * (1 - sl_roe/(100*LEVERAGE)), price_p)
        
        client.cancel_open_orders(symbol=SYMBOL)
        client.new_order(symbol=SYMBOL, side="SELL" if side=="LONG" else "BUY", positionSide=side, 
                         type="STOP_MARKET", stopPrice=sl_price, closePosition=True, workingType="MARK_PRICE")
    except: pass

def monitor_position(entry, side, qty):
    global monitoring_sides
    ai_tp = BEST_PARAMS['tp_roe']
    ai_sl = BEST_PARAMS['sl_roe']
    current_dynamic_sl_roe = -ai_sl 
    
    while True:
        try:
            # 1. Kiểm tra vị thế thực tế
            pos = get_open_positions()
            curr = next((p for p in pos if p['side'] == side), None)
            
            # Nếu vị thế không còn trên sàn, thoát Monitor ngay
            if not curr: 
                if side in monitoring_sides: monitoring_sides.remove(side)
                print(f"Vị thế {side} đã được đóng hoàn toàn.")
                break
            
            price = float(client.ticker_price(SYMBOL)["price"])
            roe = ((price - entry)/entry)*LEVERAGE*100 * (1 if side=="LONG" else -1)

            # --- CHIẾN THUẬT 3 CHẶNG ---
            if roe >= (ai_tp * 0.4) and current_dynamic_sl_roe < 1:
                current_dynamic_sl_roe = 1
                place_tp_sl(entry, side, current_dynamic_sl_roe)
            elif roe >= (ai_tp * 0.7) and current_dynamic_sl_roe < (ai_tp * 0.3):
                current_dynamic_sl_roe = round(ai_tp * 0.3, 2)
                place_tp_sl(entry, side, current_dynamic_sl_roe)
            elif roe >= (ai_tp * 0.9) and current_dynamic_sl_roe < (ai_tp * 0.6):
                current_dynamic_sl_roe = round(ai_tp * 0.6, 2)
                place_tp_sl(entry, side, current_dynamic_sl_roe)

            # --- KIỂM TRA ĐÓNG LỆNH ---
            if roe >= ai_tp or roe <= current_dynamic_sl_roe:
                reason = "🎯 CHẠM TP AI" if roe >= ai_tp else "🛡️ CHẠM SL ĐỘNG"
                send_tele(f"🚨 **ĐÓNG {side}:** {reason}\nROE: `{roe:.2f}%` - Đang thực hiện đóng...")
                
                # BƯỚC QUAN TRỌNG: Hủy toàn bộ lệnh chờ để dọn đường
                try: client.cancel_open_orders(symbol=SYMBOL)
                except: pass
                
                # Thử đóng lệnh 3 lần
                for i in range(3):
                    try:
                        # Gửi lệnh MARKET ngược chiều để đóng lệnh
                        client.new_order(
                            symbol=SYMBOL, 
                            side="SELL" if side=="LONG" else "BUY", 
                            positionSide=side, 
                            type="MARKET", 
                            quantity=curr['qty'] # Lấy đúng qty thực tế hiện tại
                        )
                        send_tele(f"✅ **ĐÃ ĐÓNG XONG {side}**")
                        break 
                    except Exception as e:
                        print(f"Lỗi đóng lệnh lần {i+1}: {e}")
                        time.sleep(1)
                
                # Sau khi đóng thành công hoặc hết lượt thử, thoát vòng lặp
                if side in monitoring_sides: monitoring_sides.remove(side)
                break 
                
            time.sleep(2)
        except Exception as e:
            print(f"Lỗi Monitor {side}: {e}")
            time.sleep(5)

def report_status_periodically():
    while True:
        try:
            acc = client.account()
            bal = next(float(a['walletBalance']) for a in acc['assets'] if a['asset'] == 'USDT')
            pos = get_open_positions()
            _, _, price_p = get_symbol_filters()
            
            msg = f"🏦 **BÁO CÁO STATUS**\n"
            msg += f"💰 Vốn: `{bal:.2f} USDT`\n"
            msg += f"⚙️ AI Params: TP `{BEST_PARAMS['tp_roe']}%` | SL `{BEST_PARAMS['sl_roe']}%`"
            msg += f"\n---"

            if pos:
                price = float(client.ticker_price(SYMBOL)["price"])
                for p in pos:
                    entry = p['entry']
                    side = p['side']
                    roe = ((price - entry)/entry)*LEVERAGE*100 * (1 if side=="LONG" else -1)
                    
                    # TÍNH GIÁ TP VÀ SL THỰC TẾ
                    tp_p = round(entry * (1 + BEST_PARAMS['tp_roe']/(100*LEVERAGE)), price_p) if side=="LONG" else round(entry * (1 - BEST_PARAMS['tp_roe']/(100*LEVERAGE)), price_p)
                    sl_p = round(entry * (1 - BEST_PARAMS['sl_roe']/(100*LEVERAGE)), price_p) if side=="LONG" else round(entry * (1 + BEST_PARAMS['sl_roe']/(100*LEVERAGE)), price_p)
                    
                    msg += f"\n🔸 **{side}**"
                    msg += f"\n   ROE: `{roe:.2f}%` | Entry: `{entry}`"
                    msg += f"\n   🎯 Giá TP: `{tp_p}`"
                    msg += f"\n   🚩 Giá SL: `{sl_p}`" # Dòng này sẽ giúp bạn biết SL đang ở đâu
            else:
                msg += "\n📭 Hiện tại không có lệnh nào."

            send_tele(msg)
            time.sleep(900)
        except Exception as e:
            print(f"Lỗi báo cáo: {e}")
            time.sleep(30)

# ====================== MAIN LOOP ======================
def main():
    global last_order_candle, monitoring_sides
    sync_time()
    send_tele("🤖 **BOT RESTARTED - OPTIMIZING...**")
    try: client.change_position_mode(dualSidePosition=True)
    except: pass

    if not os.path.exists(f"{MODEL_PATH}/models.pkl"): train_models()
    else: optimize_trading_params()
        
    models = pickle.load(open(f"{MODEL_PATH}/models.pkl", "rb"))
    Thread(target=report_status_periodically, daemon=True).start()
    last_train = datetime.datetime.now()

    while True:
        try:
            active_pos = get_open_positions()
            for p in active_pos:
                if p['side'] not in monitoring_sides:
                    monitoring_sides.add(p['side'])
                    place_tp_sl(p['entry'], p['side'])
                    Thread(target=monitor_position, args=(p['entry'], p['side'], p['qty']), daemon=True).start()

            if (datetime.datetime.now() - last_train).total_seconds() > 86400:
                train_models()
                models = pickle.load(open(f"{MODEL_PATH}/models.pkl", "rb"))
                last_train = datetime.datetime.now()

            now = datetime.datetime.now(datetime.UTC)
            candle = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
            if candle != last_order_candle:
                preds = []
                for tf in ["15m","3m","1m"]:
                    kl = client.klines(SYMBOL, tf, limit=50)
                    df = prepare_features(pd.DataFrame(kl))
                    probs = [m.predict_proba(df[FEATURES].tail(1))[0][1] for m in models]
                    avg = sum(probs)/3
                    preds.append("LONG" if avg>=0.65 else "SHORT" if avg<=0.35 else None)

                if preds.count(preds[0]) == len(preds) and preds[0] and preds[0] not in [p['side'] for p in active_pos]:
                    price = float(client.ticker_price(SYMBOL)["price"])
                    client.new_order(symbol=SYMBOL, side="BUY" if preds[0]=="LONG" else "SELL", 
                                     positionSide=preds[0], type="MARKET", quantity=calc_qty(price))
                    send_tele(f"🚀 **VÀO LỆNH {preds[0]}**")
                last_order_candle = candle
            time.sleep(10)
        except Exception as e:
            if "orderId" not in str(e): print(f"Error: {e}")
            time.sleep(15)

if __name__ == "__main__":
    main()
