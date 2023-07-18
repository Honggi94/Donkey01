import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtGui import QStandardItem
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import BinanceAPI
from mplfinance.original_flavor import candlestick2_ohlc
import PyQt5.QtGui
import subprocess
import sys


form_class = uic.loadUiType("mainwindow.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.api_key = ''
        self.api_secret = ''
        self.binance = BinanceAPI.Binance(self.api_key, self.api_secret)
        self.train_process = 0
        self.ai_statis = '인공지능 상태창\n'

        self.pushButton.clicked.connect(self.btn_clicked)
        self.pushButton_2.clicked.connect(self.limit_buy_clicked)
        self.pushButton_3.clicked.connect(self.limit_sell_clicked)
        self.pushButton_4.clicked.connect(self.market_buy_clicked)
        self.pushButton_5.clicked.connect(self.market_sell_clicked)
        self.pushButton_6.clicked.connect(self.load_now_pirce)
        self.pushButton_8.clicked.connect(self.clean_position)
        self.pushButton_9.clicked.connect(self.clean_order)
        self.pushButton_10.clicked.connect(self.train_start)
        self.pushButton_11.clicked.connect(self.train_stop)
        self.pushButton_12.clicked.connect(self.auto_on)
        self.pushButton_13.clicked.connect(self.auto_off)
        self.horizontalSlider.valueChanged.connect(self.value_change)

        # for draw graph
        self.fig = plt.Figure()
        self.fig.set_facecolor("#f0f0f0")
        self.canvas = FigureCanvas(self.fig)
        self.graph_verticalLayout.addWidget(self.canvas)

    def btn_clicked(self):
        # draw graph
        t_interval = self.comboBox_4.currentText()
        skhynix = self.binance.lookup_from_now_price(self.comboBox_3.currentText(), t_interval, 100)
        ax = self.fig.add_subplot()
        ax.set_facecolor("#f0f0f0")
        ax.clear()
        candlestick2_ohlc(ax, skhynix['open'], skhynix['high'], skhynix['low'], skhynix['close'], width=0.5, colorup='r', colordown='b')
        self.canvas.draw()

        # print price
        self.label_20.setText(str(skhynix['close'].values[99]))

        # check account money
        total, used, Available = self.binance.lookup_account()
        self.label_4.setText(str(round(float(total), 2)))
        self.label_5.setText(str(round(float(used), 2)))
        self.label_6.setText(str(round(float(Available), 2)))

        # check my position
        coin, entryPrice, positionAmt, unrealizedProfit, isolatedWallet, my_time= self.binance.lookup_position()
        total_position = "포지션\n"
        for i in range(len(coin)):
            total_position += f"코인 : {coin[i]}, 진입가격 : {entryPrice[i]}, 수량 : {round(float(positionAmt[i]), 2)}, 미실현손익 : {round(float(unrealizedProfit[i]), 2)}, 격리금액 : {round(float(isolatedWallet[i]),2)}, 거래시간 : {my_time[i]}\n"
        self.label_11.setText(total_position)

        # check my wait trade
        exist_coin, time, id, amount, price = self.binance.lookup_wait_postion()
        total_wait_tarde = "거래대기 및 오류내용\n"
        for i in range(len(time)):
            total_wait_tarde += f"{exist_coin[i]} - 거래시간 : {time[i]}, 주문번호 : {id[i]}, 수량 : {amount[i]}, 주문금액 : {price[i]}\n"
        self.label_12.setText(total_wait_tarde)

        try:
            # leverage edit
            leverage = self.binance.leverage_controll(self.comboBox_3.currentText(), self.spinBox.value())
            self.label_10.setText('x' + str(leverage))
        
        except:
            self.label_12.setText('거래대기 및 오류내용\n레버리지 오류, 보유 중인 레버리지 코인이 있습니다')


    def limit_buy_clicked(self):
        try:
            total_buy_price = float(self.lineEdit_4.text())
            buy_price = float(self.textEdit.toPlainText())
            buy_amount = total_buy_price / buy_price
            self.label_14.setText(str(buy_amount))
            self.binance.limit_buy(self.comboBox.currentText(), buy_amount, buy_price)
            self.btn_clicked()
        except:
            print('주문 수량이 맞지 않습니다')
            self.label_12.setText('거래대기 및 오류내용\n주문 수량이 너무 적습니다. 자세한 것은 API 참조')

    def limit_sell_clicked(self):
        try:
            total_sell_price = float(self.lineEdit_4.text())
            sell_price = float(self.textEdit.toPlainText())
            sell_amount = total_sell_price / sell_price
            self.label_14.setText(str(sell_amount))
            self.binance.limit_sell(self.comboBox.currentText(), sell_amount, sell_price)
            self.btn_clicked()
        except:
            print('주문 수량이 맞지 않습니다')
            self.label_12.setText('거래대기 및 오류내용\n주문 수량이 너무 적습니다. 자세한 것은 API 참조')

    def load_now_pirce(self):
        try:
            df = self.binance.lookup_from_now_price(self.comboBox.currentText(), '1m', 1)
            now_price = df['close'].values[0]
            self.textEdit.setText(str(now_price))
        except:
            print('현재가 불러오기 실패')
            self.label_12.setText('거래대기 및 오류내용\n현재가를 불러올 수 없습니다. 자세한 것은 API 참조')


    def market_buy_clicked(self):
        try:
            df = self.binance.lookup_from_now_price(self.comboBox_2.currentText(), '1m', 1)
            now_price = float(df['close'].values[0])
            total_buy_price = float(self.label_16.text())
            buy_amount = total_buy_price / now_price
            self.binance.market_buy(self.comboBox_2.currentText(), buy_amount)
            self.btn_clicked()
        except:
            print('주문 수량이 맞지 않습니다')
            self.label_12.setText('거래대기 및 오류내용\n주문 수량이 너무 적습니다. 자세한 것은 API 참조')

    def market_sell_clicked(self):
        try:
            df = self.binance.lookup_from_now_price(self.comboBox_2.currentText(), '1m', 1)
            now_price = float(df['close'].values[0])
            total_sell_price = float(self.label_16.text())
            sell_amount = total_sell_price / now_price
            self.binance.market_sell(self.comboBox_2.currentText(), sell_amount)
            self.btn_clicked()
        except:
            print('주문 수량이 맞지 않습니다')
            self.label_12.setText('거래대기 및 오류내용\n주문 수량이 너무 적습니다. 자세한 것은 API 참조')

    def value_change(self, value):
        Available = float(self.label_6.text())
        self.label_16.setText(str(Available - Available*(100-value)/100))

    def clean_position(self):
        try:
            coin, _, positionAmt, _, _, _ = self.binance.lookup_position()
            print(coin)
            for i in range(len(coin)):
                if float(positionAmt[i]) > 0:
                    self.binance.market_sell(coin[i], float(positionAmt[i]))
                if float(positionAmt[i]) < 0:
                    #coin[i][:3]+'/'+coin[i][-4:]
                    self.binance.market_buy(coin[i], -float(positionAmt[i]))
            self.btn_clicked()
        except:
            print('포지션 정리가 되지 않습니다')
            self.label_12.setText('거래대기 및 오류내용\n주문 수량이 너무 적습니다. 자세한 것은 API 참조')

    def clean_order(self):
        exist_coin, time, id, amount, price = self.binance.lookup_wait_postion()
        try:
            for i in range(len(exist_coin)):
                self.binance.cancelOrder(exist_coin[i], id[i])
            self.btn_clicked()
        except:
            print('대기중인 거래의 정리가 되지 않습니다')
            self.label_12.setText('거래대기 및 오류내용\n주문 수량이 너무 적습니다. 자세한 것은 API 참조')

    def train_start(self):
        try:
            if not "XRP 훈련 진행중..." in self.ai_statis:
                if not "Auto XRP Trading 진행중..." in self.ai_statis:
                    if "훈련 종료" in self.ai_statis:
                        self.ai_statis = '인공지능 상태창\n' + 'XRP 훈련 진행중...\n'
                        self.train_process = subprocess.Popen(args=[sys.executable, 'Donkey_Train.py'])
                        self.label_21.setText(self.ai_statis)
                    else:
                        self.ai_statis += 'XRP 훈련 진행중...\n'
                        self.train_process = subprocess.Popen(args=[sys.executable, 'Donkey_Train.py'])
                        self.label_21.setText(self.ai_statis)
                else:
                    self.ai_statis += 'auto 작동중으로 별도 훈련 불가\n'
                    self.label_21.setText(self.ai_statis)
            else:
                self.ai_statis += '이미 XRP 훈련 진행중 입니다\n'
                self.label_21.setText(self.ai_statis)
        except:
            print('API 참조')
            self.label_12.setText('거래대기 및 오류내용\ntrain start 오류. 자세한 것은 API 참조\n')

    def train_stop(self):
        try:
            if "XRP 훈련 진행중..." in self.ai_statis:
                self.ai_statis = '인공지능 상태창\n' + 'XRP 훈련 종료\n'
                self.train_process.kill()
                self.label_21.setText(self.ai_statis)
            else:
                self.ai_statis += '훈련 진행중이지 않아 훈련 종료 불가\n'
                self.label_21.setText(self.ai_statis)
        except:
            print('API 참조')
            self.label_12.setText('거래대기 및 오류내용\ntrain stop 오류. 자세한 것은 API 참조\n')

    def auto_on(self):
        try:
            if not 'Auto XRP Trading 진행중...' in self.ai_statis:
                if not "XRP 훈련 진행중" in self.ai_statis:
                    if '자동 거래 종료' in self.ai_statis:
                        self.ai_statis = '인공지능 상태창\n' + 'Auto XRP traing 진행중...\n' + 'Auto XRP Trading 진행중...\n'
                        self.auto_process = subprocess.Popen(args=[sys.executable, 'Auto_Trading.py'])
                        self.label_21.setText(self.ai_statis)
                    else:
                        self.ai_statis +='Auto XRP traing 진행중...\nAuto XRP Trading 진행중...\n'
                        self.auto_process = subprocess.Popen(args=[sys.executable, 'Auto_Trading.py'])
                        self.label_21.setText(self.ai_statis)
                else:
                    self.ai_statis += '훈련을 멈추고 Auto 기능을 사용해주세요\n'
                    self.label_21.setText(self.ai_statis)
            else:
                self.ai_statis += '이미 Auto XRP Trading 진행중입니다\n'
                self.label_21.setText(self.ai_statis)
        except:
            print('API 참조')
            self.label_12.setText('거래대기 및 오류내용\nauto start 오류. 자세한 것은 API 참조\n')

    def auto_off(self):
        try:
            if 'Auto XRP Trading 진행중...' in self.ai_statis:
                self.ai_statis = '인공지능 상태창\n'+ 'Auto XRP traing 종료\n' + 'Auto XRP Trading 종료\n'
                self.auto_process.kill()
                self.label_21.setText(self.ai_statis)
            else:
                self.ai_statis += 'Auto XRP Trading 사용중이지 않습니다\n'
                self.label_21.setText(self.ai_statis)
        except:
            print('API 참조')
            self.label_12.setText('거래대기 및 오류내용\nauto stop 오류. 자세한 것은 API 참조\n')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
