import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
from environment import StockTradingEnv
from dqn_solver import DQNAgent
from data_preprocessing import get_stock_data



def train_dqn_agent(stock_data, agent, env, num_episodes=20, batch_size=64):
    """
    Huấn luyện tác nhân DQN với dữ liệu giao dịch.
    :param stock_data: Dữ liệu cổ phiếu (dưới dạng DataFrame)
    :param agent: Tác nhân DQN
    :param env: Môi trường giao dịch chứng khoán
    :param num_episodes: Số lượng tập huấn luyện
    :param batch_size: Kích thước batch khi huấn luyện DQN
    :return: Danh sách giá trị danh mục đầu tư sau mỗi tập
    """
    portfolio_values = []  # Lưu giá trị danh mục qua các tập

    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])

        for t in range(len(stock_data)):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {episode + 1}/{num_episodes}, Total Value: {env.total_value}")
                portfolio_values.append(env.total_value)  # Lưu giá trị tài khoản
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    return portfolio_values


def plot_portfolio_values(portfolio_values):
    """
    Vẽ biểu đồ giá trị danh mục đầu tư qua các tập huấn luyện.
    :param portfolio_values: Danh sách giá trị danh mục đầu tư sau mỗi tập
    """
    plt.plot(portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value')
    plt.show()


def plot_candlestick_chart(data):
    """
    Vẽ biểu đồ nến từ dữ liệu giao dịch cổ phiếu.
    :param data: Dữ liệu cổ phiếu (bao gồm các cột Open, High, Low, Close)
    """
    data_for_chart = data[['Open', 'High', 'Low', 'Close']]
    mpf.plot(data_for_chart, type='candle', style='charles', title="Candlestick Chart", volume=True)


def main():
    # Bước 1: Lấy dữ liệu chứng khoán
    stock_data = get_stock_data('AAPL', '2020-01-01', '2023-01-01')

    # Bước 2: Khởi tạo môi trường và mô hình DQN
    env = StockTradingEnv(stock_data)
    agent = DQNAgent(env.state_size, action_size=3)  # State size và action size được lấy từ môi trường

    # Bước 3: Huấn luyện mô hình
    portfolio_values = train_dqn_agent(stock_data, agent, env)

    # Bước 4: Hiển thị kết quả giá trị danh mục đầu tư
    plot_portfolio_values(portfolio_values)

    # Bước 5: Hiển thị biểu đồ nến của cổ phiếu
    plot_candlestick_chart(stock_data)

    if len(agent.memory) > batch_size:
        print("Replaying batch...")
        agent.replay(batch_size)
        print("Batch replay done")


if __name__ == '__main__':
    main()
