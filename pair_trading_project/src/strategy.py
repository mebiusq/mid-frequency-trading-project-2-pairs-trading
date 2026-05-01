import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os


class StrategyMixin:
    def calculate_half_life(self,
                            spread):  # считаем коинтеграцию, причем нам нужна не просто нужна она, а как быстро спред вернется к 0, показывает ожидаемое время, за которое отклонение сократится вдвое.
        lag = spread.shift(1)
        ret = spread - lag

        df_temp = pd.DataFrame({'ret': ret, 'lag': lag}).dropna()
        if df_temp.empty:
            return np.inf

        ret_clean = df_temp['ret']
        lag_clean = df_temp['lag']

        if lag_clean.nunique() < 2:
            return np.inf

        X = sm.add_constant(lag_clean)
        mdl = sm.OLS(ret_clean, X).fit()

        if len(mdl.params) < 2:
            return np.inf

        theta = -mdl.params.iloc[1]
        if theta <= 0:
            return np.inf
        return -np.log(0.5) / theta

    def train_pairs(
            self):  # ищем коинтегрированные пары на трейне, учитывая есть ли долгосрочная связь и достаточно ли быстрая она
        train = self.data.loc[self.cfg['train_start']:self.cfg['valid_end']]
        cols = train.columns
        candidates = []

        for i in range(len(cols)):  # не очень эффективно, но перебираем все пары
            for j in range(i + 1, len(cols)):
                s1, s2 = train[cols[i]], train[cols[j]]

                try:
                    score, pval, _ = coint(s1, s2)  # тест, H0: нет коинтеграции
                except Exception:
                    continue

                if pval < 0.05:  # если H0 отвергается, коинтеграция есть, проверяем её скорость
                    mdl = sm.OLS(s1, sm.add_constant(s2)).fit()
                    spread = s1 - mdl.params.iloc[1] * s2
                    hl = self.calculate_half_life(spread)
                    corr = s1.pct_change().corr(s2.pct_change())

                    if hl < self.cfg['max_half_life']:  # отсеиваем пары где не достаточно быстрая связь
                        candidates.append({
                            'y': cols[i],
                            'x': cols[j],
                            'pvalue': float(pval),
                            'half_life': float(hl),
                            'correlation': float(corr) if pd.notna(corr) else np.nan
                        })

        candidates.sort(key=lambda x: (x['pvalue'], -x['correlation'] if pd.notna(
            x['correlation']) else 999))  # сортируем, потом убираем тикеры, чтобы каждый был только в 1 паре
        used = set()
        self.pairs = []
        for c in candidates:
            if c['y'] not in used and c['x'] not in used:
                self.pairs.append(c)
                used.add(c['y'])
                used.add(c['x'])
                if len(self.pairs) >= self.cfg['max_pairs']:
                    break

    def _build_pair_frame(self, y_name, x_name, start=None, end=None,
                          entry_threshold=None):  # готовим датафрейм признаков для модельки
        close_prices = self.data[[y_name, x_name]].copy()
        open_prices = self.open_data[[y_name, x_name]].copy()

        if start or end:
            close_prices = close_prices.loc[start:end]
            open_prices = open_prices.loc[start:end]

        y = close_prices[y_name]
        x = close_prices[x_name]

        cov_roll = y.rolling(window=self.cfg['rolling_window']).cov(x)  # rolling OLS для динамического подсчета беты
        var_roll = x.rolling(window=self.cfg['rolling_window']).var()
        beta = cov_roll / var_roll
        beta = beta.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        beta = beta.clip(-self.cfg['beta_cap'], self.cfg['beta_cap'])

        spread = y - beta * x  # нормализация
        mean_roll = spread.rolling(window=self.cfg['z_window']).mean()
        std_roll = spread.rolling(window=self.cfg['z_window']).std()
        z = (spread - mean_roll) / std_roll.replace(0, np.nan)  # насколько текущее расхождение аномально

        # считаются всякие будущие фичи
        prev_z = z.shift(1)  # z-score прошлого дня, чтобы понимать направление движения спреда
        z_diff = z - prev_z
        abs_z = z.abs()
        spread_volatility = spread.pct_change().rolling(20).std()
        y_volatility = y.pct_change().rolling(20).std()
        x_volatility = x.pct_change().rolling(20).std()
        returns_correlation = pd.Series(y.pct_change().corr(x.pct_change()), index=close_prices.index)
        distance_mean = (spread - mean_roll) / spread.rolling(60).std().replace(0, np.nan)

        entry_thr = self.cfg['entry_threshold'] if entry_threshold is None else entry_threshold
        signal_raw = pd.Series(0, index=close_prices.index)
        signal_raw[z < -entry_thr] = 1  # если спред слишком низко, ожидаем возврат вверх
        signal_raw[z > entry_thr] = -1  # иначе вниз

        beta_abs_val = beta.abs().fillna(1)
        weight_y = 1 / (1 + beta_abs_val)
        weight_x = beta_abs_val / (1 + beta_abs_val)

        open_to_open_y = open_prices[y_name].shift(-2) / open_prices[y_name].shift(-1) - 1  # чтобы не было look-ahead
        open_to_open_x = open_prices[x_name].shift(-2) / open_prices[x_name].shift(-1) - 1

        next_ret = signal_raw * (weight_y * open_to_open_y - np.sign(beta.fillna(
            1)) * weight_x * open_to_open_x)  # целевая доходность сделки, которую моделька должна предсказывать

        data = pd.DataFrame({
            'y': y,
            'x': x,
            'beta': beta,
            'spread': spread,
            'z': z,
            'z_prev': prev_z,
            'z_change': z_diff,
            'abs_z': abs_z,
            'spread_vol': spread_volatility,
            'y_vol': y_volatility,
            'x_vol': x_volatility,
            'ret_corr': returns_correlation,
            'dist_from_mean': distance_mean,
            'raw_signal': signal_raw,
            'w_y': weight_y,
            'w_x': weight_x,
            'o2o_y': open_to_open_y,
            'o2o_x': open_to_open_x,
            'next_pair_ret': next_ret})
        return data

    def _feature_columns(self):  # список фичей
        return ['z', 'abs_z', 'z_change', 'beta', 'spread_vol', 'y_vol', 'x_vol', 'ret_corr', 'dist_from_mean']

    def _fit_logreg_for_pair(self, pair):  # обучение модели и подбор параметров
        y_name, x_name = pair['y'], pair['x']
        full_train_df = self._build_pair_frame(y_name, x_name, self.cfg['train_start'], self.cfg['train_end'])
        best_score = -np.inf
        best_result = None
        features = self._feature_columns()

        for entry_thr in self.cfg['entry_grid']:  # перебор гипов
            train_df = self._build_pair_frame(y_name, x_name, self.cfg['train_start'], self.cfg['train_end'], entry_thr)
            valid_df = self._build_pair_frame(y_name, x_name, self.cfg['valid_start'], self.cfg['valid_end'], entry_thr)

            train_df = train_df[train_df['raw_signal'] != 0].copy()
            valid_df = valid_df[valid_df['raw_signal'] != 0].copy()

            train_df = train_df.dropna(subset=features + ['next_pair_ret'])
            valid_df = valid_df.dropna(subset=features + ['next_pair_ret'])

            y_train = (train_df['next_pair_ret'] > 0).astype(
                int)  # бинарная классификация: будет сделка прибыльной или нет

            pipeline = Pipeline([  # масштабирование + дисбаланс классов
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(
                    C=0.5,
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=42
                ))
            ])

            pipeline.fit(train_df[features], y_train)
            valid_probs = pipeline.predict_proba(valid_df[features])[
                :, 1]  # вероятность того, что следующий трейд будет прибыльным

            for prob_thr in self.cfg['prob_threshold_grid']:
                for size_floor in self.cfg['size_floor_grid']:
                    take_trade = (valid_probs >= prob_thr).astype(
                        float)  # торгуем только если модель достаточно уверена
                    confidence = (valid_probs - prob_thr) / (1.0 - prob_thr)
                    confidence = np.clip(confidence, 0, 1)
                    position_size = size_floor + (
                            1 - size_floor) * confidence  # чем выше уверенность модели, тем больше размер позиции

                    pair_returns = take_trade * position_size * valid_df['next_pair_ret'].values
                    trades_num = int(take_trade.sum())

                    trade_signal = pd.Series(take_trade, index=valid_df.index)  # 0 если сделку не берем, 1 если берем
                    position_change = trade_signal.diff().abs()
                    position_change = position_change.fillna(trade_signal.abs())
                    costs = position_change * self.cfg['transaction_cost']  # считаем итоговые транзакционные издержки

                    net_returns = pd.Series(pair_returns, index=valid_df.index) - costs
                    equity_curve = (1.0 + net_returns.fillna(0)).cumprod()  # строим кривую капитала на валидации

                    total_return = equity_curve.iloc[-1] - 1.0
                    drawdowns = equity_curve / equity_curve.cummax() - 1.0
                    max_drawdown = drawdowns.min()
                    sharpe_ratio = 0.0 if net_returns.std() == 0 else np.sqrt(
                        252) * net_returns.mean() / net_returns.std()
                    hit_rate = float((valid_df['next_pair_ret'][take_trade.astype(
                        bool)] > 0).mean()) if trades_num > 0 else 0  # доля прибыльных сделок среди тех, которые модель пропустила

                    score = total_return - 0.5 * abs(max_drawdown) + 0.05 * sharpe_ratio + 0.02 * min(trades_num,
                                                                                                      25)  # балансируем доходность, просадку, качество и достаточное число сделок, формулка подбиралась так, чтобы хорошо подобрать гипы по этой формуле =)
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'model': pipeline,
                            'prob_threshold': float(prob_thr),
                            'size_floor': float(size_floor),
                            'entry_threshold': float(entry_thr),
                            'trades': trades_num,
                            'hit_rate': round(hit_rate, 4)
                        }

        if best_result is None:  # если нормальный подбор не дал результата, то базовый вариант
            df = full_train_df[full_train_df['raw_signal'] != 0].copy()
            df = df.dropna(subset=features + ['next_pair_ret'])

            if len(df) >= 20 and (df['next_pair_ret'] > 0).astype(int).nunique() == 2:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('logreg', LogisticRegression(
                        C=0.5,
                        max_iter=1000,
                        class_weight='balanced',
                        random_state=42
                    ))
                ])
                pipeline.fit(df[features], (df['next_pair_ret'] > 0).astype(int))

                best_result = {
                    'model': pipeline,
                    'prob_threshold': 0.55,
                    'size_floor': 0.50,
                    'entry_threshold': self.cfg['entry_threshold'],
                    'trades': int(len(df)),
                    'hit_rate': float((df['next_pair_ret'] > 0).mean())
                }
            else:  # если даже так обучить нельзя, модельку для этой пары не используем
                best_result = {
                    'model': None,
                    'prob_threshold': 0.55,
                    'size_floor': 0.50,
                    'entry_threshold': self.cfg['entry_threshold'],
                    'trades': 0,
                    'hit_rate': 0.0
                }

        return best_result

    def fit_models(self):  # подбор параметров для каждой пары
        self.models = {}
        self.filters = {}

        if not self.cfg['use_logreg_overlay']:
            return
        for pair in self.pairs:
            fit_result = self._fit_logreg_for_pair(pair)
            key = (pair['y'], pair['x'])
            self.models[key] = fit_result['model']
            self.filters[key] = {
                'trades': fit_result['trades'],
                'hit_rate': fit_result['hit_rate'],
                'prob_threshold': fit_result['prob_threshold'],
                'size_floor': fit_result['size_floor'],
                'entry_threshold': fit_result['entry_threshold']
            }
            print(f"{pair['y']} vs {pair['x']}: {self.filters[key]}")

        os.makedirs('artifacts', exist_ok=True)
        if self.models:
            joblib.dump(self.models, 'artifacts/signals_generator.json')
