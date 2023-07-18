import ccxt
import pandas as pd
from datetime import datetime
from binance.client import Client
import time
from time import strftime

class Binance:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.binance = ccxt.binance(config={
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })

    def lookup_account(self):
        print("**************my account***************")
        account = self.binance.fetch_balance()
        print(f'free : {account["USDT"]["free"]}\nused : {account["USDT"]["used"]}\ntotal : {account["USDT"]["total"]}')

        return account["USDT"]["total"], account["USDT"]["used"], account["USDT"]["free"],


    def lookup_from_now_price(self, symbol, timeframe, limit):
        btc2 = self.binance.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=None,
            limit=limit)

        df = pd.DataFrame(btc2, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.set_index('datetime', inplace=True)
        print("**************from now price data***************")
        print(df)

        return df

    def market_buy(self, symbol, amount):
        order = self.binance.create_market_buy_order(
            symbol=symbol, # "BTC/USDT"
            amount=amount # 0.001
        )
        print("**************market_buy***************")
        print(f"buy coin : {order['symbol']}\nbuy cost : {order['cost']}\nbuy amount : {order['amount']}\nbuy price average : {order['average']}\nbuy ID : {order['clientOrderId']}")

    def market_sell(self, symbol, amount):
        order = self.binance.create_market_sell_order(
            symbol=symbol, # "BTC/USDT"
            amount=amount # 0.001
        )
        print("**************market_sell***************")
        print(
            f"sell coin : {order['symbol']}\nsell cost : {order['cost']}\nsell amount : {order['amount']}\nsell price average : {order['average']}\nsell ID : {order['clientOrderId']}")

    def limit_buy(self, symbol, amount, price):
        order = self.binance.create_limit_buy_order(
            symbol=symbol, # "BTC/USDT"
            amount=amount, # 0.001
            price=price
        )
        print("**************limit_buy***************")
        print(f"buy coin : {order['symbol']}\nbuy cost : {order['cost']}\nbuy amount : {order['amount']}\nbuy price average : {order['average']}\nbuy ID : {order['clientOrderId']}")

    def limit_sell(self, symbol, amount, price):
        order = self.binance.create_limit_sell_order(
            symbol=symbol, # "BTC/USDT"
            amount=amount, # 0.001
            price=price
        )
        print("**************limit_sell***************")
        print(
            f"sell coin : {order['symbol']}\nsell cost : {order['cost']}\nsell amount : {order['amount']}\nsell price average : {order['average']}\nsell ID : {order['clientOrderId']}")

    def leverage_controll(self, symbol, leverage):
        markets = self.binance.load_markets()
        # symbol = "BTC/USDT"
        # leverage = 2
        market = self.binance.market(symbol)

        resp = self.binance.fapiPrivate_post_leverage({
            'symbol': market['id'],
            'leverage': leverage
        })

        print(f"leverage controlled coin : {resp['symbol']}\nleverage : {resp['leverage']}")

        return resp['leverage']

    def lookup_position(self):
        balance = self.binance.fetch_balance()
        positions = balance['info']['positions']
        coin = []
        entryPrice = []
        positionAmt = []
        unrealizedProfit = []
        isolatedWallet = []
        my_time = []
        print(positions)
        for position in positions:
            if int(float(position["positionAmt"])) is not 0:
                coin.append(position['symbol'])
                find_time = int(position['updateTime'][:10])
                t_local = time.localtime(find_time)
                time_format = '%Y-%m-%d %H:%M:%S'
                time_str = strftime(time_format, t_local)
                entryPrice.append(position['entryPrice'])
                positionAmt.append(position['positionAmt'])
                unrealizedProfit.append(position['unrealizedProfit'])
                isolatedWallet.append(position['isolatedWallet'])
                my_time.append(time_str)
                print("**************position***************")
                print(f"coin : {position['symbol']}\nentry price : {position['entryPrice']}\npostion amount : {position['positionAmt']}\nunrealized Profit : {position['unrealizedProfit']}\nisolated Wallet : {position['isolatedWallet']}\n time : {time_str}")


        return coin, entryPrice, positionAmt, unrealizedProfit, isolatedWallet, my_time

    def lookup_wait_postion(self):
        coin_list = ["BTC/USDT", "XRP/USDT", "TRX/USDT"]
        exist_coin = []
        time = []
        id = []
        amount = []
        price = []
        for coin in coin_list:
            try:
                open_orders = self.binance.fetch_open_orders(
                    symbol=coin)
                print(open_orders)
                open_orders = open_orders[0]
                print(open_orders)
                print("**************wait position***************")
                print(f"{coin} - wait postion open time : {open_orders['datetime']}\norderID : {open_orders['id']}\norder amount : {open_orders['amount']}\nprice : {open_orders['price']}\n")
                exist_coin.append(coin)
                time.append(open_orders['datetime'])
                id.append(open_orders['id'])
                amount.append(open_orders['amount'])
                price.append(open_orders['price'])
            except:
                print("no wait postion")

        return exist_coin, time, id, amount, price

    def cancelOrder(self, symbol, orderid):
        resp  = self.binance.cancel_order(orderid, symbol)
        print(resp)


if __name__ == '__main__':
    api_key = ''
    api_secret = ''
    binance = Binance(api_key, api_secret)
    binance.lookup_position()

