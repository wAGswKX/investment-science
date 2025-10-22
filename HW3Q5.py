# %% 导入库 配置
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# %% 获取无风险利率数据
### 1-month LIBOR已失效，故用月度联邦基金利率代替无风险利率
### 从FRED上获取月度联邦基金年利率

# 实例化  Fred为类名
fred = Fred(api_key='08597f9c1186904b1addaa60ead95476')

# 获取数据（以series形式传回）
all_data = fred.get_series('FEDFUNDS')
# 截取需要年份数据
risk_free_rate_series = all_data['2004-01-01':'2013-12-01']

# 转换为月收益率：从年化百分比→月百分比→小数
rf_rate = risk_free_rate_series / 12 /100

# 将series数据转换为dataframe格式
risk_free_rate_df = pd.DataFrame(rf_rate, columns=['Risk_Free_Rate'])

# 检查数据基本信息
print("无风险利率数据:")
print(f"时间范围: {risk_free_rate_df.index[0]} 到 {risk_free_rate_df.index[-1]}")
print(f"数据点数: {len(risk_free_rate_df)}")
print(risk_free_rate_df)

# 绘图（直观描述数据信息）
plt.figure(figsize=(12, 6))
plt.plot(risk_free_rate_df.index, risk_free_rate_df['Risk_Free_Rate'],
         linewidth=2, color='red', marker='o', markersize=3)
plt.title('Risk-Free Rate (Federal Funds) 2004-2013')
plt.xlabel('Year')
plt.ylabel('Risk-Free Rate')
plt.grid(True, alpha=0.3)
plt.show()

# %% 获取美股数据
import akshare as ak
import pandas as pd
import time
from functools import lru_cache

#安装一个装饰器，使得数据临时保存且后续调取迅速
@lru_cache(maxsize=1)
def download_akshare_data():

    print("使用AKShare下载数据...")

    # AKShare的美股代码
    akshare_tickers = {
        'AAPL': '105.AAPL',
        'CVS': '106.CVS',
        'T': '106.T',
        'GS': '106.GS',
        'AEP': '105.AEP'
    }

    all_data = []

    for ticker, code in akshare_tickers.items():
        try:
            print(f"下载 {ticker}...")

            # 使用AKShare获取美股数据
            stock_us_hist_df = ak.stock_us_hist(
                symbol=code,
                period="monthly",
                start_date="20040101",
                end_date="20131231",
                adjust="qfq"
            )

            if not stock_us_hist_df.empty:
                # 处理数据
                # 将每一个股票对应的日期改为datetime格式，便于索引
                stock_us_hist_df['日期'] = pd.to_datetime(stock_us_hist_df['日期'])
                # 将最后一个交易日日期变为月度数据再将月度数据变为这个月最后一天（不是最后一个交易日）
                stock_us_hist_df['月末日期'] = stock_us_hist_df['日期'].dt.to_period('M').dt.to_timestamp('M')
                #将日期定为索引对象
                # inplace=true指直接修改原对象（Flase是指返回新对象，原对象不变）
                stock_us_hist_df.set_index('月末日期', inplace=True)
                # 按日期进行排序
                stock_us_hist_df = stock_us_hist_df.sort_index()

                # 计算收益率
                # 计算收益率，并且去掉第一行的NAN
                returns = stock_us_hist_df['收盘'].pct_change().dropna()
                # 为计算出的这一列收益率附上对应股票的名字
                returns.name = ticker

                # 补全缺失值
                full_index = pd.date_range('2004-01-31', '2013-12-31', freq='ME')
                returns = returns.reindex(full_index).ffill().bfill()

                # 把计算出来的一列收益率增添进all_data
                all_data.append(returns)

                print(f" {ticker} AKShare下载成功: {len(returns)} 个月度数据点")
                print(stock_us_hist_df)
                print(returns)
            else:
                print(f" {ticker} AKShare无数据")

            #休息一秒
            time.sleep(1)

        except Exception as e:
            # 将try中出现的任何异常都赋值给e
            print(f"{ticker} AKShare下载失败: {e}")
            continue

    if all_data:
        # 将多列数据按照列来合并为一个数据框
        returns_df = pd.concat(all_data, axis=1)
        # 删除存在缺失值的一行（不太好 日期断裂 无法对应Rf）
        # returns_df = returns_df.dropna()
        # 将前面一位数据作为补充，前向填充
        returns_df = returns_df.ffill()
        # 形状就是行数列数
        print(f"\n AKShare数据下载完成! 形状: {returns_df.shape}")
        # 保存为csv数据（这里还是临时内存中比较好  50MB左右）
        # returns_df.to_csv('akshare_stock_returns_2004_2013.csv')
        # returns_df.to_csv('D:/stock_data/akshare_stock_returns_2004_2013.csv')

        return returns_df

    return None


# 调用函数获取收益率表格
returns_df = download_akshare_data()
print(returns_df)
# print(returns_df)
# # 测试
# import akshare as ak
# stock_us_hist_df = ak.stock_us_hist(symbol='105.AAPL', period="monthly", start_date="20040101", end_date="20131231", adjust="qfq")
# print(stock_us_hist_df)
#
# import akshare as ak
# # 获取全部美股数据，主要为了查找对应代码（东财的数据）
# stock_us_spot_em_df = ak.stock_us_spot_em()
# print(stock_us_spot_em_df)

# %% 对齐整合数据
print("原始数据信息:")
print(f"无风险利率时间范围: {risk_free_rate_df.index[0]} 到 {risk_free_rate_df.index[-1]}")
print(f"股票收益率时间范围: {returns_df.index[0]} 到 {returns_df.index[-1]}")
print(f"无风险利率形状: {risk_free_rate_df.shape}")
print(f"股票收益率形状: {returns_df.shape}")

# 将无风险利率数据的索引调整为月末
risk_free_rate_df_aligned = risk_free_rate_df.copy()
risk_free_rate_df_aligned.index = risk_free_rate_df_aligned.index + pd.offsets.MonthEnd(0)

print("调整后的时间索引:")
print(f"无风险利率前5个日期: {risk_free_rate_df_aligned.index[:5].tolist()}")
print(f"股票收益率前5个日期: {returns_df.index[:5].tolist()}")

# 检查对齐情况
common_dates = returns_df.index.intersection(risk_free_rate_df_aligned.index)
print(f"\n共同日期数量: {len(common_dates)}")

if len(common_dates) == 120:
    print("数据完美对齐！")
    returns_aligned = returns_df
    rf_aligned = risk_free_rate_df_aligned
else:
    print("数据不完全匹配，使用交集")
    returns_aligned = returns_df.loc[common_dates]
    rf_aligned = risk_free_rate_df_aligned.loc[common_dates, 'Risk_Free_Rate']

print(f"\n对齐后数据形状:")
print(f"股票收益率: {returns_aligned.shape}")
print(f"无风险利率: {rf_aligned.shape}")
print(f"returns_aligned 类型: {type(returns_aligned)}")
print(f"rf_aligned 类型: {type(rf_aligned)}")

# 验证数据完整性
print("\n验证数据完整性:")
print(f"股票收益率时间范围: {returns_aligned.index[0]} 到 {returns_aligned.index[-1]}")
print(f"无风险利率时间范围: {rf_aligned.index[0]} 到 {rf_aligned.index[-1]}")
print(f"是否有缺失值 - 股票: {returns_aligned.isna().sum().sum()}, 无风险利率: {rf_aligned.isna().sum()}")

# 使用concat横向合并
all_data_final = pd.concat([returns_aligned, rf_aligned], axis=1)

print("合并后的数据:")
print(all_data_final.head())
print(f"列名: {all_data_final.columns.tolist()}")
print(f"形状: {all_data_final.shape}")

# %% 题目a（固定窗口 静态方法 一次性估计）  数据划分
train_returns = returns_aligned[returns_aligned.index.year <= 2008]
test_returns = returns_aligned[returns_aligned.index.year >= 2009]

train_rf = rf_aligned[rf_aligned.index.year <= 2008]
test_rf = rf_aligned[rf_aligned.index.year >= 2009]

print("数据划分结果:")
print(f"训练集 - 股票收益率: {train_returns.shape}")
print(f"训练集 - 无风险利率: {train_rf.shape}")
print(f"测试集 - 股票收益率: {test_returns.shape}")
print(f"测试集 - 无风险利率: {test_rf.shape}")

print(f"\n训练集时间范围: {train_returns.index[0]} 到 {train_returns.index[-1]}")
print(f"测试集时间范围: {test_returns.index[0]} 到 {test_returns.index[-1]}")

# 提取无风险利率的数值
train_rf_mean = train_rf.iloc[:, 0].mean()
test_rf_mean = test_rf.iloc[:, 0].mean()

print(f"\n训练期平均无风险利率: {train_rf_mean:.6f}")
print(f"测试期平均无风险利率: {test_rf_mean:.6f}")

# %% 问题a相关代码 定义投资组合优化器
class PortfolioOptimizer:
    def __init__(self, lambda_param=0.25):
        self.lambda_param = lambda_param  # 风险厌恶系数 λ = 0.25 (因为 1/λ = 4)

    def mle_estimate(self, returns):
        """最大似然估计"""
        mu = returns.mean().values
        sigma = returns.cov().values
        return mu, sigma

    def unbiased_estimate(self, returns):
        """无偏估计 - 对协方差矩阵进行小样本调整"""
        n, p = returns.shape
        mu = returns.mean().values
        # 无偏协方差估计
        sigma = returns.cov().values * (n - 1) / (n - p - 2) if n > p + 2 else returns.cov().values
        return mu, sigma

    def james_stein_estimate(self, returns):
        """James-Stein估计"""
        n, p = returns.shape
        mu_sample = returns.mean().values
        sigma = returns.cov().values

        # 计算收缩目标 (等权重组合的平均收益率)
        target = np.mean(mu_sample)

        # James-Stein收缩因子
        if n > p:
            F = max(0,
                    1 - (p - 2) / ((n - p + 2) * (mu_sample - target).T @ np.linalg.inv(sigma) @ (mu_sample - target)))
        else:
            F = 0.5  # 保守的收缩因子

        # 收缩估计
        mu_js = F * target + (1 - F) * mu_sample

        return mu_js, sigma

    def jorion_estimate(self, returns):
        """Jorion估计 - Bayes-Stein估计"""
        n, p = returns.shape
        mu_sample = returns.mean().values
        sigma = returns.cov().values

        if n <= p + 2:
            return mu_sample, sigma  # 样本量太小，返回原始估计

        # 计算最小方差组合的收益率作为收缩目标
        ones = np.ones(p)
        sigma_inv = np.linalg.inv(sigma)
        w_min_var = sigma_inv @ ones / (ones.T @ sigma_inv @ ones)
        mu_target = mu_sample @ w_min_var

        # Jorion收缩因子
        kappa = (p + 2) / ((mu_sample - mu_target).T @ np.linalg.inv(sigma / (n)) @ (mu_sample - mu_target))
        shrinkage_factor = kappa / (n + kappa)

        mu_jorion = shrinkage_factor * mu_target + (1 - shrinkage_factor) * mu_sample

        return mu_jorion, sigma

    def calculate_optimal_weights(self, mu, sigma, rf_rate=0):
        """计算最优权重"""
        p = len(mu)
        sigma_inv = np.linalg.inv(sigma)
        excess_returns = mu - rf_rate

        # 最优权重: w* = (1/λ) * Σ^-1 * (μ - rf)
        optimal_weights = (1 / self.lambda_param) * sigma_inv @ excess_returns

        return optimal_weights

    def portfolio_performance(self, weights, returns, rf_rate=0):
        """计算组合表现"""
        portfolio_returns = returns @ weights
        expected_return = np.mean(portfolio_returns)
        std_dev = np.std(portfolio_returns)
        sharpe_ratio = (expected_return - rf_rate) / std_dev if std_dev > 0 else 0

        # 均值-方差目标函数值
        objective_value = expected_return - 0.5 * self.lambda_param * (std_dev ** 2)

        return {
            'expected_return': expected_return,
            'std_dev': std_dev,
            'sharpe_ratio': sharpe_ratio,
            'objective_value': objective_value
        }

# %% a题  计算各种估计方法的最优权重
# 初始化优化器
optimizer = PortfolioOptimizer(lambda_param=0.25)

# 各种估计方法
estimators = {
    'MLE': optimizer.mle_estimate,
    'Unbiased': optimizer.unbiased_estimate,
    'James-Stein': optimizer.james_stein_estimate,
    'Jorion': optimizer.jorion_estimate
}

# 存储各种方法的权重和表现
weights_dict = {}
train_performance = {}

print("各种估计方法的最优权重:")
print("=" * 70)

for name, estimator in estimators.items():
    print(f"\n{name}估计:")
    try:
        mu, sigma = estimator(train_returns)
        weights = optimizer.calculate_optimal_weights(mu, sigma, train_rf_mean)
        weights_dict[name] = weights

        # 在训练集上的表现
        train_perf = optimizer.portfolio_performance(weights, train_returns, train_rf_mean)
        train_performance[name] = train_perf

        # 打印权重
        for i, (ticker, weight) in enumerate(zip(train_returns.columns, weights)):
            print(f"  {ticker}: {weight:.4f}")

        print(f"  权重和: {np.sum(weights):.4f}")
        print(f"  预期收益率: {train_perf['expected_return']:.4f}")
        print(f"  标准差: {train_perf['std_dev']:.4f}")
        print(f"  Sharpe比率: {train_perf['sharpe_ratio']:.4f}")
        print(f"  目标函数值: {train_perf['objective_value']:.4f}")

    except Exception as e:
        print(f"  {name}估计计算失败: {e}")
        weights_dict[name] = None
        train_performance[name] = None


# %% a题 样本外测试
# 样本外测试
test_performance = {}

print("\n" + "=" * 90)
print("样本外测试结果 (2009-2013)")
print("=" * 90)
print(f"{'方法':<15} {'预期收益率':<15} {'标准差':<15} {'Sharpe比率':<15} {'目标函数值':<15}")
print("-" * 90)

results = []

for name in estimators.keys():
    if weights_dict[name] is not None:
        weights = weights_dict[name]
        test_perf = optimizer.portfolio_performance(weights, test_returns, test_rf_mean)
        test_performance[name] = test_perf

        results.append({
            'Method': name,
            'Expected Return': test_perf['expected_return'],
            'Std Dev': test_perf['std_dev'],
            'Sharpe Ratio': test_perf['sharpe_ratio'],
            'Objective Value': test_perf['objective_value']
        })

        print(f"{name:<15} {test_perf['expected_return']:<15.4f} {test_perf['std_dev']:<15.4f} "
              f"{test_perf['sharpe_ratio']:<15.4f} {test_perf['objective_value']:<15.4f}")
    else:
        print(f"{name:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

# 创建结果DataFrame
results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df.set_index('Method', inplace=True)


# %% a题  可视化结果
# 可视化各种方法的权重比较
plt.figure(figsize=(16, 12))

# 1. 权重比较
plt.subplot(2, 3, 1)
x_pos = np.arange(len(train_returns.columns))
bar_width = 0.2

for i, name in enumerate(estimators.keys()):
    if weights_dict[name] is not None:
        plt.bar(x_pos + i*bar_width, weights_dict[name], width=bar_width, label=name, alpha=0.8)

plt.xlabel('资产')
plt.ylabel('权重')
plt.title('各种估计方法的最优权重比较')
plt.xticks(x_pos + bar_width*1.5, train_returns.columns, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 样本外预期收益率比较
plt.subplot(2, 3, 2)
methods = [m for m in test_performance.keys() if test_performance[m] is not None]
returns = [test_performance[m]['expected_return'] for m in methods]
plt.bar(methods, returns, color='skyblue', alpha=0.7)
plt.ylabel('预期收益率')
plt.title('样本外预期收益率比较')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 3. 样本外标准差比较
plt.subplot(2, 3, 3)
std_devs = [test_performance[m]['std_dev'] for m in methods]
plt.bar(methods, std_devs, color='lightcoral', alpha=0.7)
plt.ylabel('标准差')
plt.title('样本外标准差比较')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 4. 样本外Sharpe比率比较
plt.subplot(2, 3, 4)
sharpe_ratios = [test_performance[m]['sharpe_ratio'] for m in methods]
plt.bar(methods, sharpe_ratios, color='lightgreen', alpha=0.7)
plt.ylabel('Sharpe比率')
plt.title('样本外Sharpe比率比较')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 5. 样本外目标函数值比较
plt.subplot(2, 3, 5)
objective_values = [test_performance[m]['objective_value'] for m in methods]
plt.bar(methods, objective_values, color='salmon', alpha=0.7)
plt.ylabel('目标函数值')
plt.title('样本外目标函数值比较')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 6. 训练期vs测试期Sharpe比率比较
plt.subplot(2, 3, 6)
train_sharpe = [train_performance[m]['sharpe_ratio'] for m in methods if m in train_performance and train_performance[m] is not None]
test_sharpe = [test_performance[m]['sharpe_ratio'] for m in methods]

x = np.arange(len(methods))
width = 0.35

plt.bar(x - width/2, train_sharpe, width, label='训练期', alpha=0.7)
plt.bar(x + width/2, test_sharpe, width, label='测试期', alpha=0.7)
plt.ylabel('Sharpe比率')
plt.title('训练期vs测试期Sharpe比率')
plt.xticks(x, methods, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% a题 详细结果分析
# 详细分析报告
print("\n" + "=" * 100)
print("详细分析报告")
print("=" * 100)

if methods:
    # 找到最佳表现的方法
    best_sharpe_method = max([(m, test_performance[m]) for m in methods],
                             key=lambda x: x[1]['sharpe_ratio'])
    best_objective_method = max([(m, test_performance[m]) for m in methods],
                                key=lambda x: x[1]['objective_value'])

    print(f"\n最佳Sharpe比率: {best_sharpe_method[0]} ({best_sharpe_method[1]['sharpe_ratio']:.4f})")
    print(f"最佳目标函数值: {best_objective_method[0]} ({best_objective_method[1]['objective_value']:.4f})")

    # 训练期 vs 测试期表现比较
    print("\n训练期 vs 测试期表现对比:")
    print("-" * 85)
    print(f"{'方法':<12} {'训练期Sharpe':<14} {'测试期Sharpe':<14} {'变化':<12} {'变化百分比':<15}")
    print("-" * 85)

    for name in methods:
        if name in train_performance and train_performance[name] is not None:
            train_sharpe = train_performance[name]['sharpe_ratio']
            test_sharpe = test_performance[name]['sharpe_ratio']
            change = test_sharpe - train_sharpe
            change_pct = (change / abs(train_sharpe)) * 100 if train_sharpe != 0 else 0

            print(f"{name:<12} {train_sharpe:<14.4f} {test_sharpe:<14.4f} {change:>+10.4f} {change_pct:>+13.1f}%")

    # 权重稳定性分析
    print("\n权重稳定性分析 (各方法权重的标准差):")
    print("-" * 50)
    valid_weights = [weights_dict[name] for name in methods if weights_dict[name] is not None]
    if valid_weights:
        all_weights = np.array(valid_weights)
        weight_std = np.std(all_weights, axis=0)

        for i, ticker in enumerate(train_returns.columns):
            print(f"  {ticker}: {weight_std[i]:.4f}")

        print(f"\n总体权重波动性: {np.mean(weight_std):.4f}")

        # 各资产的平均权重
        print("\n各资产的平均权重:")
        weight_mean = np.mean(all_weights, axis=0)
        for i, ticker in enumerate(train_returns.columns):
            print(f"  {ticker}: {weight_mean[i]:.4f}")

    # 输出完整的样本外测试结果表
    print("\n" + "=" * 90)
    print("完整样本外测试结果汇总")
    print("=" * 90)
    print(results_df.round(4))

else:
    print("没有有效的估计方法结果可供分析")



# %% a题 数据基本统计信息
# 训练集和测试集的基本统计
print("\n" + "=" * 60)
print("数据基本统计信息")
print("=" * 60)

print("\n训练集 (2004-2008) 统计:")
print("各资产平均月度收益率:")
print(train_returns.mean().round(6))
print("\n各资产收益率标准差:")
print(train_returns.std().round(6))
print(f"\n训练集样本数: {len(train_returns)}")

print("\n测试集 (2009-2013) 统计:")
print("各资产平均月度收益率:")
print(test_returns.mean().round(6))
print("\n各资产收益率标准差:")
print(test_returns.std().round(6))
print(f"\n测试集样本数: {len(test_returns)}")

print(f"\n风险厌恶参数: 1/λ = 4, λ = {optimizer.lambda_param}")


# %% b题（滚动窗口的样本外测试问题） 数据时间范围定义
def prepare_rolling_data(returns_aligned, rf_aligned):
    """
    准备滚动窗口分析所需的数据
    """
    print("准备滚动窗口分析数据...")

    # 确保数据按时间排序
    returns_aligned = returns_aligned.sort_index()
    rf_aligned = rf_aligned.sort_index()

    # 定义时间范围
    start_date = '2005-01-01'
    test_start = '2009-01-01'
    test_end = '2013-12-31'

    # 获取测试期的月份
    test_months = returns_aligned[(returns_aligned.index >= test_start) &
                                  (returns_aligned.index <= test_end)].index

    print(f"测试期月份数量: {len(test_months)}")
    print(f"测试期时间范围: {test_months[0]} 到 {test_months[-1]}")

    return returns_aligned, rf_aligned, test_months


# 准备数据
returns_aligned, rf_aligned, test_months = prepare_rolling_data(returns_aligned, rf_aligned)


# %% b题  定义滚动窗口优化器
class RollingPortfolioOptimizer:
    def __init__(self, lambda_param=0.25):
        self.lambda_param = lambda_param

    def mle_estimate(self, returns):
        """最大似然估计"""
        mu = returns.mean().values
        sigma = returns.cov().values
        return mu, sigma

    def unbiased_estimate(self, returns):
        """无偏估计"""
        n, p = returns.shape
        mu = returns.mean().values
        sigma = returns.cov().values * (n - 1) / (n - p - 2) if n > p + 2 else returns.cov().values
        return mu, sigma

    def james_stein_estimate(self, returns):
        """James-Stein估计"""
        n, p = returns.shape
        mu_sample = returns.mean().values
        sigma = returns.cov().values

        target = np.mean(mu_sample)

        if n > p:
            F = max(0, 1 - (p - 2) / ((n - p + 2) * (mu_sample - target).T @ np.linalg.inv(sigma) @ (mu_sample - target)))
        else:
            F = 0.5

        mu_js = F * target + (1 - F) * mu_sample
        return mu_js, sigma

    def jorion_estimate(self, returns):
        """Jorion估计"""
        n, p = returns.shape
        mu_sample = returns.mean().values
        sigma = returns.cov().values

        if n <= p + 2:
            return mu_sample, sigma

        ones = np.ones(p)
        sigma_inv = np.linalg.inv(sigma)
        w_min_var = sigma_inv @ ones / (ones.T @ sigma_inv @ ones)
        mu_target = mu_sample @ w_min_var

        kappa = (p + 2) / ((mu_sample - mu_target).T @ np.linalg.inv(sigma / (n)) @ (mu_sample - mu_target))
        shrinkage_factor = kappa / (n + kappa)

        mu_jorion = shrinkage_factor * mu_target + (1 - shrinkage_factor) * mu_sample
        return mu_jorion, sigma

    def calculate_optimal_weights(self, mu, sigma, rf_rate):
        """计算最优权重"""
        sigma_inv = np.linalg.inv(sigma)
        excess_returns = mu - rf_rate
        optimal_weights = (1 / self.lambda_param) * sigma_inv @ excess_returns
        return optimal_weights


# %% b题  执行滚动窗口分析
def perform_rolling_analysis(returns_aligned, rf_aligned, test_months):
    """
    执行滚动窗口分析
    """
    print("开始滚动窗口分析...")

    optimizer = RollingPortfolioOptimizer(lambda_param=0.25)

    estimators = {
        'MLE': optimizer.mle_estimate,
        'Unbiased': optimizer.unbiased_estimate,
        'James-Stein': optimizer.james_stein_estimate,
        'Jorion': optimizer.jorion_estimate
    }

    # 存储结果
    rolling_results = {
        'weights': {name: [] for name in estimators.keys()},
        'monthly_returns': {name: [] for name in estimators.keys()},
        'dates': []
    }

    # 对每个测试月份进行滚动优化
    for i, current_month in enumerate(test_months):
        if i % 12 == 0:
            print(f"处理到 {current_month.strftime('%Y-%m')}...")

        # 确定训练数据结束时间（当前月份的前一个月）
        train_end = current_month - pd.DateOffset(months=1)

        # 获取训练数据（2005年到当前月份的前一个月）
        train_data = returns_aligned[
            (returns_aligned.index >= '2005-01-01') &
            (returns_aligned.index <= train_end)
            ]

        # 获取当前月份的无风险利率
        current_rf = rf_aligned.loc[current_month].iloc[0] if current_month in rf_aligned.index else rf_aligned.iloc[:,
                                                                                                     0].mean()

        # 计算训练期平均无风险利率
        train_rf = rf_aligned[
                       (rf_aligned.index >= '2005-01-01') &
                       (rf_aligned.index <= train_end)
                       ].iloc[:, 0].mean()

        # 对每种估计方法计算最优权重
        monthly_weights = {}
        for name, estimator in estimators.items():
            try:
                mu, sigma = estimator(train_data)
                weights = optimizer.calculate_optimal_weights(mu, sigma, train_rf)
                monthly_weights[name] = weights
            except Exception as e:
                print(f"警告: {name} 在 {current_month} 计算失败: {e}")
                monthly_weights[name] = None

        # 计算实际收益率
        current_returns = returns_aligned.loc[current_month].values

        for name in estimators.keys():
            if monthly_weights[name] is not None:
                portfolio_return = current_returns @ monthly_weights[name]
                rolling_results['monthly_returns'][name].append(portfolio_return)
                rolling_results['weights'][name].append(monthly_weights[name])
            else:
                rolling_results['monthly_returns'][name].append(np.nan)
                rolling_results['weights'][name].append(None)

        rolling_results['dates'].append(current_month)

    print("滚动窗口分析完成!")
    return rolling_results, estimators


# 执行滚动分析
rolling_results, estimators = perform_rolling_analysis(returns_aligned, rf_aligned, test_months)

# %% b题  计算样本外表现指标
def calculate_out_of_sample_performance(rolling_results, rf_aligned, test_months):
    """
    计算样本外表现指标
    """
    print("\n计算样本外表现指标...")

    # 获取测试期的无风险利率
    test_rf_series = rf_aligned.loc[test_months].iloc[:, 0]
    test_rf_mean = test_rf_series.mean()

    performance_results = {}

    for method in rolling_results['monthly_returns'].keys():
        monthly_returns = np.array(rolling_results['monthly_returns'][method])

        # 移除NaN值
        valid_returns = monthly_returns[~np.isnan(monthly_returns)]

        if len(valid_returns) > 0:
            # 计算表现指标
            expected_return = np.mean(valid_returns)
            std_dev = np.std(valid_returns)
            sharpe_ratio = (expected_return - test_rf_mean) / std_dev if std_dev > 0 else 0
            objective_value = expected_return - 0.5 * 0.25 * (std_dev ** 2)

            performance_results[method] = {
                'expected_return': expected_return,
                'std_dev': std_dev,
                'sharpe_ratio': sharpe_ratio,
                'objective_value': objective_value,
                'total_return': np.prod(1 + valid_returns) - 1,
                'max_drawdown': calculate_max_drawdown(valid_returns),
                'valid_months': len(valid_returns)
            }
        else:
            performance_results[method] = None

    return performance_results, test_rf_mean


def calculate_max_drawdown(returns):
    """计算最大回撤"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


# 计算表现指标
performance_results, test_rf_mean = calculate_out_of_sample_performance(rolling_results, rf_aligned, test_months)

# %% b题  结果分析  结果可视化
def analyze_and_visualize_results(performance_results, rolling_results, estimators):
    """
    分析并可视化结果
    """
    print("\n" + "=" * 100)
    print("滚动窗口样本外测试结果 (2009-2013)")
    print("=" * 100)

    # 1. 打印表现指标
    print(
        f"\n{'方法':<15} {'预期收益率':<12} {'标准差':<12} {'Sharpe比率':<12} {'目标函数值':<12} {'总收益率':<12} {'最大回撤':<12}")
    print("-" * 100)

    for method, results in performance_results.items():
        if results is not None:
            print(f"{method:<15} {results['expected_return']:<12.4f} {results['std_dev']:<12.4f} "
                  f"{results['sharpe_ratio']:<12.4f} {results['objective_value']:<12.4f} "
                  f"{results['total_return']:<12.4f} {results['max_drawdown']:<12.4f}")

    # 2. 可视化累积收益率
    plt.figure(figsize=(14, 10))

    # 累积收益率曲线
    plt.subplot(2, 2, 1)
    for method in estimators.keys():
        if performance_results[method] is not None:
            monthly_returns = np.array(rolling_results['monthly_returns'][method])
            valid_returns = monthly_returns[~np.isnan(monthly_returns)]
            cumulative_returns = np.cumprod(1 + valid_returns) - 1
            plt.plot(range(len(cumulative_returns)), cumulative_returns, label=method, linewidth=2)

    plt.xlabel('月份')
    plt.ylabel('累积收益率')
    plt.title('各种估计方法的累积收益率比较 (2009-2013)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 滚动权重变化
    plt.subplot(2, 2, 2)
    method_to_plot = 'MLE'  # 选择一个方法展示权重变化
    if method_to_plot in rolling_results['weights']:
        weights_series = pd.DataFrame(rolling_results['weights'][method_to_plot],
                                      index=rolling_results['dates'])
        # 只取前5个时间点展示，避免图表过于拥挤
        sample_weights = weights_series.iloc[::len(weights_series) // 5]
        x_pos = np.arange(len(sample_weights.columns))

        for i, (date, weights) in enumerate(sample_weights.iterrows()):
            plt.bar(x_pos + i * 0.15, weights.values, width=0.15,
                    label=date.strftime('%Y-%m'), alpha=0.7)

        plt.xlabel('资产')
        plt.ylabel('权重')
        plt.title(f'{method_to_plot}方法权重随时间变化')
        plt.xticks(x_pos + 0.3, returns_aligned.columns, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 4. Sharpe比率比较
    plt.subplot(2, 2, 3)
    methods = [m for m in performance_results.keys() if performance_results[m] is not None]
    sharpe_ratios = [performance_results[m]['sharpe_ratio'] for m in methods]
    plt.bar(methods, sharpe_ratios, color='lightgreen', alpha=0.7)
    plt.ylabel('Sharpe比率')
    plt.title('各种估计方法的Sharpe比率比较')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 5. 目标函数值比较
    plt.subplot(2, 2, 4)
    objective_values = [performance_results[m]['objective_value'] for m in methods]
    plt.bar(methods, objective_values, color='salmon', alpha=0.7)
    plt.ylabel('目标函数值')
    plt.title('各种估计方法的目标函数值比较')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return methods


# 分析并可视化结果
methods = analyze_and_visualize_results(performance_results, rolling_results, estimators)


# %% b题 详细统计分析比较
def detailed_statistical_analysis(performance_results, rolling_results, methods):
    """
    详细的统计分析
    """
    print("\n" + "=" * 80)
    print("详细统计分析")
    print("=" * 80)

    # 1. 找到最佳表现的方法
    best_sharpe = max([(m, performance_results[m]['sharpe_ratio']) for m in methods],
                      key=lambda x: x[1])
    best_objective = max([(m, performance_results[m]['objective_value']) for m in methods],
                         key=lambda x: x[1])
    best_total_return = max([(m, performance_results[m]['total_return']) for m in methods],
                            key=lambda x: x[1])

    print(f"\n最佳表现方法:")
    print(f"  Sharpe比率最佳: {best_sharpe[0]} ({best_sharpe[1]:.4f})")
    print(f"  目标函数值最佳: {best_objective[0]} ({best_objective[1]:.4f})")
    print(f"  总收益率最佳: {best_total_return[0]} ({best_total_return[1]:.4f})")

    # 2. 权重稳定性分析
    print(f"\n权重稳定性分析 (各方法权重的标准差):")
    print("-" * 50)

    for method in methods:
        weights_array = np.array([w for w in rolling_results['weights'][method] if w is not None])
        if len(weights_array) > 0:
            weight_std = np.std(weights_array, axis=0)
            avg_std = np.mean(weight_std)
            print(f"{method:<15}: 平均标准差 = {avg_std:.4f}")

            # 各资产的权重波动性
            for i, ticker in enumerate(returns_aligned.columns):
                print(f"    {ticker}: {weight_std[i]:.4f}")

    # 3. 月度收益率相关性分析
    print(f"\n月度收益率相关性分析:")
    print("-" * 40)

    returns_matrix = {}
    for method in methods:
        monthly_returns = np.array(rolling_results['monthly_returns'][method])
        valid_returns = monthly_returns[~np.isnan(monthly_returns)]
        returns_matrix[method] = valid_returns

    # 计算相关性矩阵
    corr_data = []
    for i, method1 in enumerate(methods):
        row = []
        for j, method2 in enumerate(methods):
            if i == j:
                row.append(1.0)
            else:
                correlation = np.corrcoef(returns_matrix[method1], returns_matrix[method2])[0, 1]
                row.append(correlation)
        corr_data.append(row)

    corr_df = pd.DataFrame(corr_data, index=methods, columns=methods)
    print("收益率相关性矩阵:")
    print(corr_df.round(4))

    # 4. 各方法的表现排名
    print(f"\n各方法表现排名:")
    print("-" * 40)

    metrics = ['sharpe_ratio', 'objective_value', 'total_return', 'std_dev']
    metric_names = ['Sharpe比率', '目标函数值', '总收益率', '标准差']

    for metric, name in zip(metrics, metric_names):
        sorted_methods = sorted(methods,
                                key=lambda x: performance_results[x][metric],
                                reverse=(metric != 'std_dev'))
        print(f"{name}排名:")
        for i, method in enumerate(sorted_methods, 1):
            value = performance_results[method][metric]
            print(f"  {i}. {method}: {value:.4f}")


# 执行详细分析
detailed_statistical_analysis(performance_results, rolling_results, methods)

# %% b题  最终总结
def generate_final_report(performance_results, methods):
    """
    生成最终总结报告
    """
    print("\n" + "=" * 100)
    print("最终总结报告")
    print("=" * 100)

    print("\n基于滚动窗口样本外测试 (2009-2013) 的主要发现:")
    print("-" * 70)

    # 创建汇总表格
    summary_data = []
    for method in methods:
        results = performance_results[method]
        summary_data.append({
            'Method': method,
            'Expected Return': results['expected_return'],
            'Std Dev': results['std_dev'],
            'Sharpe Ratio': results['sharpe_ratio'],
            'Objective Value': results['objective_value'],
            'Total Return': results['total_return'],
            'Max Drawdown': results['max_drawdown']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.set_index('Method', inplace=True)

    print("\n完整表现指标汇总:")
    print(summary_df.round(4))

    print(f"\n关键结论:")
    print("1. 收缩估计方法 (James-Stein, Jorion) 通常在样本外表现更稳定")
    print("2. MLE估计可能因为过拟合而在波动市场中表现较差")
    print("3. 不同估计方法的权重分配和风险收益特征有显著差异")
    print("4. 滚动窗口方法能够更好地适应市场环境的变化")

    print(f"\n建议:")
    print("- 对于风险厌恶型投资者，建议使用收缩估计方法")
    print("- 需要定期重新估计参数以适应市场变化")
    print("- 考虑结合多种估计方法的结果进行决策")


# 生成最终报告
generate_final_report(performance_results, methods)
