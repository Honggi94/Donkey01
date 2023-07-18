from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse
import os
import random
import requests
from datetime import datetime
import time
from tqdm import tqdm
import h5py
import sys
sys.path.append("..")
from Bitcoin import BinanceAPI
import time

class Donkey:
    def __init__(self, stock_data, data_len):
        self.stock_data = stock_data
        self.stock_scaler_pointer = 0
        self.stock_data_pointer = 0
        self.data_len = data_len
        self.syn_data = []
        self.input_data = []
        self.change_data = []

    def data_generator(self):
        self.generate_moving_average()
        self.conversion_to_change()

        df = self.stock_data[0]['close'].values.tolist()
        self.change_data.append(df[-self.data_len:])

        self.stock_scaler_pointer, self.stock_data_pointer = self.min_max_scale(self.stock_data, 1)


        self.syn_data.append(self.synthesis_data(self.stock_data_pointer))


        return self.syn_data, self.change_data



    def generate_moving_average(self):
        for i in range(len(self.stock_data)):
            df = self.stock_data[i].copy()
            df['MA10'] = df['close'].rolling(window=10).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA40'] = df['close'].rolling(window=40).mean()
            df['MA80'] = df['close'].rolling(window=80).mean()
            df = df[79:]
            self.stock_data[i] = df


    def conversion_to_change(self):
        scale_cols_stock = ['open', 'high', 'low', 'close', 'volume', 'MA10', 'MA20', 'MA40', 'MA80']
        for i in range(len(self.stock_data)):
            if len(self.stock_data[i]) is not 0:
                for col_name in scale_cols_stock:
                    self.stock_data[i][col_name] = self.stock_data[i][col_name].pct_change() * 100
                df = self.stock_data[i]
                df = df.replace([np.inf, -np.inf, 100, -100,], np.nan)
                df = df.dropna()
                df = df[-self.data_len:]
                self.stock_data[i] = df


    def min_max_scale(self, data, stockorindicator):
        scaler_pointer = []
        data_pointer = []
        if stockorindicator == 1:
            scale_cols = ['open', 'high', 'low', 'close', 'volume',  'MA10', 'MA20', 'MA40', 'MA80']
        if stockorindicator == 2:
            scale_cols = ['close', 'MA20']
        for i in range(len(data)):
            if len(data[i]) == 0:
                scaler_pointer.append([])
                data_pointer.append([])
            else:
                scaler = MinMaxScaler()
                mid_data = scaler.fit_transform(data[i][scale_cols])
                scaler_pointer.append(scaler)
                data_pointer.append(mid_data)
        return scaler_pointer, data_pointer

    def inverse_scale(self, scalar, input_data, stockorindicator):
        if stockorindicator == 1:
            scale_cols = ['open', 'high', 'low', 'close','volume', 'MA10', 'MA20', 'MA40', 'MA80']
        data = scalar.inverse_transform(input_data)
        data = pd.DataFrame(data)
        data.columns = scale_cols
        return data


    def synthesis_data(self, data):
        syn_len = len(data)
        syn_data = data[0]
        for i in range(1, syn_len):
            syn_data = np.concatenate((syn_data, data[i]), axis=1)
        return syn_data

    def make_input_data(self, data):
        input_data = []
        if len(data) == 0:
            return []
        else:
            for i in range(len(data)):
                input_data.append(np.reshape(data[i], [-1, ]))
        return input_data

class DQNAgent:
    def __init__(self, state_space, episodes=20):
        self.action_space = [0, 1]
        self.memory = []
        self.gamma = 0.001
        self.epsilon = 0.01
        self.epsilon_min = 0.01
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** (1. / float(episodes))

        self.buy_weights_file = 'dqn_buy_%s_%d.h5'
        self.sell_weights_file = 'dqn_sell_%s_%d.h5'
        self.seed_buy_weights_file = 'dqn_buy.h5'
        self.seed_sell_weights_file = 'dqn_buy.h5'
        n_inputs = len(state_space[0])
        n_outputs = len(self.action_space)
        self.q_model = self.build_model(n_inputs, n_outputs)
        self.q_model.compile(loss='mse', optimizer=Adam(learning_rate=1e-4))
        self.target_q_model = self.build_model(n_inputs, n_outputs)
        self.update_weights()
        self.replay_counter = 0

    def build_model(self, n_inputs, n_outputs):
        inputs = Input(shape=(n_inputs, ), name='state')
        x = Dense(128, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(n_outputs, activation='linear', name='action')(x)
        q_model = Model(inputs, x)
        q_model.summary()
        return q_model

    def save_weights(self, buy_or_sell, stock_code, i):
        if buy_or_sell == 'buy':
            self.q_model.save_weights(self.buy_weights_file % (stock_code, i))
        if buy_or_sell == 'sell':
            self.q_model.save_weights(self.sell_weights_file % (stock_code, i))

    def load_weights(self, buy_or_sell, stock_code, i):
        if buy_or_sell == 'buy':
            self.q_model.load_weights(self.buy_weights_file % (stock_code, i))
        if buy_or_sell == 'sell':
            self.q_model.load_weights(self.sell_weights_file % (stock_code, i))

    def load_seed_weights(self, buy_or_sell):
        if buy_or_sell == 'buy':
            self.q_model.load_weights(self.seed_buy_weights_file)
        if buy_or_sell == 'sell':
            self.q_model.load_weights(self.seed_sell_weights_file)

    def update_weights(self):
        self.target_q_model.set_weights(self.q_model.get_weights())

    def act(self, state, printf = 0):
        if np.random.rand() < self.epsilon:
            return random.choice(self.action_space)

        q_values = self.q_model.predict(state)
        action = np.argmax(q_values[0])
        return action

    def remember(self, state, q_value, action, reward, next_state, done):
        item = (state, q_value, action, reward, next_state, done)
        self.memory.append(item)

    def memory_reset(self):
        self.memory = []

    def get_target_q_value(self, next_state, reward):
        q_value = np.amax(self.target_q_model.predict(next_state)[0])

        q_value *= self.gamma
        q_value += reward
        return q_value

    def replay(self, batch_size, verbose =0):
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        for state, q_value, action, reward, next_state, done in sars_batch:
            q_values = self.q_model.predict(state)

            q_values[0][action] = reward if done else q_value

            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=batch_size,
                         epochs=1,
                         verbose=verbose)

        #self.update_epsilon()

        if self.replay_counter % 4 == 0:
            self.update_weights()

        self.replay_counter += 1
        self.memory_reset()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DDQNAgent(DQNAgent):
    def __init__(self,
                 state_space,
                 episodes=25):
        super().__init__(state_space,
                         episodes)

        self.buy_weights_file = 'ddqn_buy_%s_%d.h5'
        self.sell_weights_file = 'ddqn_sell_%s_%d.h5'
        self.seed_buy_weights_file = 'ddqn_buy.h5'
        self.seed_sell_weights_file = 'ddqn_buy.h5'
        print("-------------DDQN-------------")

    def get_target_q_value(self, next_state, reward):
        action = np.argmax(self.q_model.predict(next_state)[0])
        q_value = self.target_q_model.predict(next_state)[0][action]

        q_value *= self.gamma
        q_value += reward
        return q_value

def buy_env_step(pre_asset, action, position, change):
    if action == 0:
        if position == 1:
            asset = pre_asset
            asset = asset + asset * change / 100
        if position == -1:
            asset = pre_asset
            asset = asset - asset * change / 100
    elif action == 1:
        if position == 1:
            asset = pre_asset
            asset = asset + asset * change / 100
        if position == -1:
            asset = pre_asset
            asset = asset + asset * change / 100 - asset * 0.0006
        position = 1
    reward = (action-0.5)*change
    return reward, asset, position

def sell_env_step(pre_asset, action, position, change):
    if action == 0:
        if position == 1:
            asset = pre_asset
            asset = asset + asset * change / 100
        if position == -1:
            asset = pre_asset
            asset = asset - asset * change /100
    elif action == 1:
        if position == 1:
            asset = pre_asset
            asset = asset - asset * change / 100 - asset * 0.0006
        if position == -1:
            asset = pre_asset
            asset = asset - asset * change / 100
        position = -1

    reward = (0.5-action) * change
    return reward, asset, position

def env_step(pre_asset, action, change):
    if action == 1:
        asset = pre_asset
        asset = asset + pre_asset * action * change /100
    return asset

class Train():
    def __init__(self, test_period, stock_list_length, buy_agent, sell_agent, env, change, stock_code):
        self.stock_list_length = stock_list_length
        self.buy_agent = buy_agent
        self.sell_agent = sell_agent
        self.env = env
        self.change = change
        self.test_period = test_period
        self.stock_code = stock_code

    def do_train(self, saveornot_asset, final_change, final_env, loop_num):
        performance = [0] * 3
        vs_performance = [0] * 3
        saveornot3 = [0] * 3
        pre_buy_action = [0] * 3
        pre_sell_action = [0] * 3
        final_buy_action = [0] * 3
        final_sell_action = [0] * 3
        for i in range(3):
            if sum(self.change[0]) > 0:
                position = 1
            else:
                position = -1
            asset = 1
            vs_asset = 1
            done = False
            total_buy_reward = 0
            total_sell_reward = 0
            verbose = 0
            batch_size = 1
            file_path = './ddqn_buy_%s_%d.h5' % (self.stock_code, i)
            if os.path.isfile(file_path):
                self.buy_agent[i].load_weights('buy', self.stock_code, i)
                self.sell_agent[i].load_weights('sell', self.stock_code, i)
                print('load weights!')
            else:
                print('fail load weights')

            for step in range(self.stock_list_length - 1):
                state = self.env[0][step]
                state_size = len(state)
                state = np.reshape(state, [1, state_size])
                next_state = self.env[0][step+1]
                next_state = np.reshape(next_state, [1, state_size])
                r_change = self.change[0][step+1]
                if position == -1:

                    action = self.sell_agent[i].act(state)
                    vs_sell_reward, _, vs_sell_position = sell_env_step(asset, action, position, r_change)
                    if vs_sell_position == -1:
                        q_value = self.sell_agent[i].get_target_q_value(next_state, vs_sell_reward)
                        self.sell_agent[i].remember(state, q_value, action, vs_sell_reward, next_state, done)
                    if vs_sell_position == 1:
                        q_value = self.buy_agent[i].get_target_q_value(next_state, vs_sell_reward)
                        self.sell_agent[i].remember(state, q_value, action, vs_sell_reward, next_state, done)

                    action = self.buy_agent[i].act(state, 1)
                    buy_reward, asset, position = buy_env_step(asset, action, position, r_change)
                    if position == -1:
                        q_value = self.buy_agent[i].get_target_q_value(next_state, buy_reward)
                        self.buy_agent[i].remember(state, q_value, action, buy_reward, next_state, done)
                    if position == 1:
                        q_value = self.sell_agent[i].get_target_q_value(next_state, buy_reward)
                        self.buy_agent[i].remember(state, q_value, action, buy_reward, next_state, done)
                    total_buy_reward += buy_reward

                elif position == 1:
                    # print('sell position and asset = ', asset)

                    action = self.buy_agent[i].act(state)
                    vs_buy_reward, _, vs_buy_position = buy_env_step(asset, action, position, r_change)
                    if vs_buy_position == -1:
                        q_value = self.buy_agent[i].get_target_q_value(next_state, vs_buy_reward)
                        self.buy_agent[i].remember(state, q_value, action, vs_buy_reward, next_state, done)
                    if vs_buy_position == 1:
                        q_value = self.sell_agent[i].get_target_q_value(next_state, vs_buy_reward)
                        self.buy_agent[i].remember(state, q_value, action, vs_buy_reward, next_state, done)

                    action = self.sell_agent[i].act(state, 1)
                    sell_reward, asset, position = sell_env_step(asset, action, position, r_change)
                    if position == -1:
                        q_value = self.sell_agent[i].get_target_q_value(next_state, sell_reward)
                        self.sell_agent[i].remember(state, q_value, action, sell_reward, next_state, done)
                    if position == 1:
                        q_value = self.buy_agent[i].get_target_q_value(next_state, sell_reward)
                        self.sell_agent[i].remember(state, q_value, action, sell_reward, next_state, done)
                    total_sell_reward += sell_reward

                vs_asset = env_step(vs_asset, 1, r_change)

                if len(self.buy_agent[i].memory) >= batch_size:
                    self.buy_agent[i].replay(batch_size, verbose)

                if len(self.sell_agent[i].memory) >= batch_size:
                    self.sell_agent[i].replay(batch_size, verbose)

            state = self.env[0][self.stock_list_length-1]
            state_size = len(state)
            state = np.reshape(state, [1, state_size])
            pre_buy_action[i] = self.buy_agent[i].act(state)
            pre_sell_action[i] = self.sell_agent[i].act(state)

            final_state = final_env
            final_env_sie = len(final_env)
            final_state = np.reshape(final_state, [1, final_env_sie])
            final_buy_action[i] = self.buy_agent[i].act(final_state)
            final_sell_action[i] = self.sell_agent[i].act(final_state)


            saveornot1 = pre_buy_action[i] * final_change
            saveornot2 = - pre_sell_action[i] * final_change
            saveornot3[i] = saveornot1 + saveornot2

            print("XRP, loop_num %d, weight_num %d, asset = %.5f, vs_asset = %.5f, total sell reward = %.2f, total buy reward = %.2f" % (
            loop_num, i, asset, vs_asset, total_sell_reward, total_buy_reward))

            self.sell_agent[i].update_epsilon()
            self.buy_agent[i].update_epsilon()

            performance[i] = asset
            vs_performance[i] = vs_asset
            print(final_change)


        return saveornot_asset, performance, vs_performance, saveornot3, final_buy_action, final_sell_action, pre_buy_action, pre_sell_action


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-d', '--ddqn', action='store_true', help='Use Double DQN')
    args = parser.parse_args()

    api_key = ''
    api_secret = ''
    binance = BinanceAPI.Binance(api_key, api_secret)

    sum_asset = 0
    sum_vs_asset = 0
    sum_first_vs_asset = 0
    best_gain = 0
    saveornot_sum = [0] * 3
    position = 0
    total_gain = 0
    best_index = []

    coin_list = ['XRP/USDT', 'BTC/USDT', 'TRX/USDT']
    real_coin_data = []

    for coin in coin_list:
        real_coin_data.append(binance.lookup_from_now_price(coin, '5m', 6000))

    for loop_num in range(5000):

        coin_data = []
        for i in range(3):
            coin_data.append(real_coin_data[i][:300+loop_num])

        print(np.shape(coin_data))

        donkey = Donkey(coin_data, 101)
        env, change = donkey.data_generator()

        final_change = change[0][100]
        final_env = env[0][100]
        print(final_change)

        env[0] = env[0][:100]
        change[0] = change[0][:100]

        if loop_num == 0:
            buy_agent = []
            sell_agent = []
            for i in range(3):
                buy_agent.append(DDQNAgent(env[0]))
                sell_agent.append(DDQNAgent(env[0]))

        stock_list_length = len(env[0])
        test_period = 10

        coin_name = coin_list[0][:3]

        train = Train(test_period, stock_list_length, buy_agent, sell_agent, env, change, coin_name)
        episode_count = 100
        saveornot_asset = [0]*3

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

        for i in range(3):
            saveornot3_real_sum[i] += saveornot3[i]

        best_index = []
        for i in range(3):
            if saveornot3[i] > 0:
                best_index.append(i)

        if best_index != []:
            if position == -1:
                if final_buy_action[best_index[len(best_index)-1]] == 1:
                    position = 1
            if position == 1:
                if final_sell_action[best_index[len(best_index)-1]] == 1:
                    position = -1
            if position == 0:
                if final_buy_action[best_index[len(best_index)-1]] == 1 and final_sell_action[best_index[len(best_index)-1]] == 0:
                    position = 1
                if final_buy_action[best_index[len(best_index)-1]] == 0 and final_sell_action[best_index[len(best_index)-1]] == 1:
                    position = -1

        for i in range(3):
            saveornot_sum[i] += saveornot3[i]

        print(performance)
        print(vs_performance)
        sum_asset += sum(performance)
        sum_vs_asset += sum(vs_performance)

        print('&&&&&&&&&&&&&&&&&&&&&&&&')
        print(sum_asset)
        print(sum_vs_asset)
        print(saveornot3)
        print(saveornot_sum)
        print('total gain = ', total_gain)
        print('position = ', position)






