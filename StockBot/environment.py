import numpy as np

class StockTradingEnv:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_value = initial_balance
        self.state_size = 3

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        return self._next_observation()

    def _next_observation(self):
        current_price = self.data['Close'].values[self.current_step]
        obs = np.array([current_price, self.balance, self.shares_held])
        return obs

    def step(self, action):
        current_price = self.data['Close'].values[self.current_step]

        # Hành động 1 là "Mua"
        if action == 1:
            if self.balance > current_price:
                self.shares_held += 1
                self.balance -= current_price

        # Hành động 2 là "Bán"
        elif action == 2:
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price

        # Tăng bước hiện tại
        self.current_step += 1
        self.total_value = self.balance + self.shares_held * current_price

        # Kết thúc nếu đã hết dữ liệu
        done = self.current_step >= len(self.data) - 1
        reward = self.total_value - self.initial_balance

        return self._next_observation(), reward, done
