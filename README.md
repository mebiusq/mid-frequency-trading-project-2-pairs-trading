### Структура проекта

```bash
pair_trading_project/
│
├── pair_trading_project/         # Основной код проекта
│   ├── artifacts/
│   │   ├── signals_generator.json  # Бинарник генератора сигналов стратегии, создается в src/strategy
│   │   ├── backtest_metrics.json   # Бэктестовые метрики стратегии, создается в src/run_backtest
│   │   ├── requirements_plot.json  # График требований к обеспечению вашей стратегии и динамики свободных денежных средств в формате PNG, создается в src/run_backtest
│   │   └── pnl_plot.png            # График PnL стратегии в формате PNG, создается в src/run_backtest
│   ├── data/
│   │   ├── train_data.parquet      # Котировки обучающей выборки, создаются в src/get_data
│   │   └── backtest_data.parquet   # Котировки бэктеста, создаются в src/get_data
│   └── src/                    # Исходный код
│       ├── ArbitragePortfolio.py   # Код бэктестера стратегий парного трейдинга
│       ├── get_data.py         # Качаем данные
│       ├── run_backtest.py     # Инференс стратегии
│       ├── strategy.py         # Правила торговли стратегии
│       └── utils.py            # Вспомогательные модули
├── main.py                     # Основной файл проекта, запускающий цикл стратегии от обучения до оценки метрик
├── config.yaml                 # Конфиг с требованиями к бэктесту
├── pyproject.toml              # Конфиг проекта
│
└── README.md                  
```
