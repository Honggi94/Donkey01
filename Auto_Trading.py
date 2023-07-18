import BinanceAPI
import Donkey_Train
import os
import time


api_key = ''
api_secret = ''
binance = BinanceAPI.Binance(api_key, api_secret)

if __name__ == '__main__':
    initial_total, _, _ = binance.lookup_account()
    total_gain = 0
    sum_asset = 0
    sum_vs_asset = 0
    loop_num = 1
    has_amount = 0
    saveornot3_real_sum = [0]*3
    saveornot3_sum = 0
    position = 0
    while(1):
        now = time.localtime()
        now_min = now.tm_min % 5
        if now_min < 4:
            print(now_min)
            time.sleep(10)
            continue

        coin_list = ['XRP/USDT', 'BTC/USDT', 'TRX/USDT']
        coin_data = []
        for coin in coin_list:
            coin_data.append(binance.lookup_from_now_price(coin, '5m', 400))

        print("coin data")
        print(coin_data)

        donkey = Donkey_Train.Donkey(coin_data, 101)
        env, change = donkey.data_generator()

        final_change = change[0][100]
        final_env = env[0][100]


        env[0] = env[0][:100]
        change[0] = change[0][:100]

        if loop_num == 1:
            buy_agent = []
            sell_agent = []
            for i in range(3):
                buy_agent.append(Donkey_Train.DDQNAgent(env[0]))
                sell_agent.append(Donkey_Train.DDQNAgent(env[0]))

        stock_list_length = len(env[0])
        test_period = 10

        coin_name = coin_list[0][:3]

        train = Donkey_Train.Train(test_period, stock_list_length, buy_agent, sell_agent, env, change, coin_name)
        saveornot_asset = [0] * 3

        total_gain += final_change * position

        if final_change * position > 0:
            for index in best_index:
                buy_agent[index].save_weights('buy', coin_name, index)
                sell_agent[index].save_weights('sell', coin_name, index)
                print(index, " save weights")

        for i in range(3):
            file_path = './ddqn_buy_%s_%d.h5' % (coin_name, i)
            if os.path.isfile(file_path):
                buy_agent[i].load_weights('buy', coin_name, i)
                sell_agent[i].load_weights('sell', coin_name, i)
                print('load weights!')
            else:
                print('fail load weights')


        saveornot_asset, performance, vs_performance, saveornot3, final_buy_action, final_sell_action, pre_buy_action, pre_sell_action = train.do_train(saveornot_asset, final_change, final_env, loop_num)

        loop_num += 1

        for i in range(3):
            saveornot3_real_sum[i] += saveornot3[i]

        best_index = []
        for i in range(3):
            if saveornot3[i] > 0:
                best_index.append(i)

        max_index = -1
        if best_index != []:
            max_index = best_index[len(best_index) - 1]

        if max_index != -1:
            real_buy_action = final_buy_action[max_index]
            real_sell_action = final_sell_action[max_index]
            amount_p = 0
            coin, _, positionAmt, _, _, _ = binance.lookup_position()
            if coin:
                amount_p = float(positionAmt[0])
                if amount_p > 0:
                    ("long position")
                else:
                    ("short postion")
            if amount_p < 0:
                if real_buy_action == 1:
                    total, used, Available = binance.lookup_account()
                    df = binance.lookup_from_now_price('XRP/USDT', '5m', 1)
                    now_price = df['close'].values[0]
                    order_price = now_price + now_price/50
                    order_amount = (total / 1.3)/now_price - amount_p
                    binance.limit_buy('XRP/USDT', order_amount, order_price)
                    position = 1
                else:
                    print("no trade")
            if amount_p > 0:
                if real_sell_action == 1:
                    total, used, Available = binance.lookup_account()
                    df = binance.lookup_from_now_price('XRP/USDT', '5m', 1)
                    now_price = df['close'].values[0]
                    order_price = now_price - now_price / 50
                    order_amount = (total / 1.3)/now_price + amount_p
                    binance.limit_sell('XRP/USDT', order_amount, order_price)
                    position = -1
                else:
                    print("no trade")
            if amount_p == 0:
                if real_buy_action == 0 and real_sell_action == 1:
                    total, used, Available = binance.lookup_account()
                    df = binance.lookup_from_now_price('XRP/USDT', '5m', 1)
                    now_price = df['close'].values[0]
                    order_price = now_price - now_price / 50
                    order_amount = (total / 1.3) / now_price + amount_p
                    binance.limit_sell('XRP/USDT', order_amount, order_price)
                    position = -1
                elif real_buy_action == 1 and real_sell_action == 0:
                    total, used, Available = binance.lookup_account()
                    df = binance.lookup_from_now_price('XRP/USDT', '5m', 1)
                    now_price = df['close'].values[0]
                    order_price = now_price + now_price/50
                    order_amount = (total / 1.3)/now_price - amount_p
                    binance.limit_buy('XRP/USDT', order_amount, order_price)
                    position = 1
                else:
                    print("no trade")
        exist_coin, _, id, _, _ = binance.lookup_wait_postion()
        if len(exist_coin) != 0:
            for i in range(len(exist_coin)):
                binance.cancelOrder(exist_coin[i], id[i])

        print(performance)
        print(vs_performance)
        sum_asset += sum(performance)
        sum_vs_asset += sum(vs_performance)
        print(pre_buy_action)
        print(pre_sell_action)
        print(final_buy_action)
        print(final_sell_action)
        print('&&&&&&&&&&&&&&&&&&&&&&&&')
        print(sum_asset)
        print(sum_vs_asset)
        print(saveornot3)
        print(saveornot3_real_sum)
        print(saveornot3_sum)
        print('------------------------')
        print('total gain = ', total_gain)
        print('position = ', position)

        total, used, Available = binance.lookup_account()
        if total < initial_total - initial_total/5:
            break
