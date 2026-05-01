import warnings

warnings.filterwarnings("ignore")

from pair_trading_project.src.utils import params  # Подтягиваем параметры отсюда
from pair_trading_project.src.get_data import DataFetcherMixin
from pair_trading_project.src.strategy import StrategyMixin
from pair_trading_project.src.run_backtest import BacktestMixin


class RobustPairsTrading(DataFetcherMixin, StrategyMixin, BacktestMixin):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = None
        self.open_data = None
        self.pairs = []
        self.results = {}
        self.models = {}
        self.filters = {}
        self.equity = None
        self.daily_rets = None


if __name__ == "__main__":
    strat = RobustPairsTrading(params)
    strat.fetch_data()
    strat.train_pairs()
    print("\nSelected pairs:")
    for p in strat.pairs:
        print(p)
    strat.fit_models()
    strat.backtest()
    strat.report()
