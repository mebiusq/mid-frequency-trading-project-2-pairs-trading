import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os


class BacktestMixin:
    def backtest(
            self):  # вместо фиксированного коэффициента, будем пересчитывать бета каждое окно, чтобы учитывать изменения рынка
        test_idx = self.data.loc[self.cfg["test_start"]:self.cfg["test_end"]].index
        pair_exposures = {}

        for p in self.pairs:
            key = (p["y"], p["x"])
            entry_thr = self.filters.get(key, {}).get("entry_threshold", self.cfg["entry_threshold"])

            df = self._build_pair_frame(
                p["y"], p["x"],
                self.cfg["test_start"], self.cfg["test_end"],
                entry_thr
            ).loc[test_idx].copy()

            pos = 0.0
            desired = np.zeros(len(df))
            z_arr = df["z"].values
            raw_arr = df["raw_signal"].values

            for i in range(len(z_arr)):  # разные проверки
                zval = z_arr[i]
                raw = raw_arr[i]

                if np.isnan(zval):
                    pos = 0.0
                    desired[i] = 0.0
                    continue
                if abs(zval) > self.cfg["stop_loss"]:
                    pos = 0.0
                elif pos == 0:  # вне рынка
                    if raw == 1:
                        pos = 1.0  # шорт
                    elif raw == -1:
                        pos = -1.0  # лонг
                elif pos != 0:  # в рынке
                    if abs(zval) < self.cfg["exit_threshold"]:
                        pos = 0.0  # возврат к среднему
                    if pos == 1.0 and zval > entry_thr:
                        pos = -1.0
                    if pos == -1.0 and zval < -entry_thr:
                        pos = 1.0

                desired[i] = pos

            desired = pd.Series(desired, index=df.index)
            executed = desired.copy()
            if self.cfg["use_logreg_overlay"]:
                feat_columns = self._feature_columns()
                model = self.models.get(key)
                filt = self.filters.get(key)

                if model is not None and filt is not None:
                    valid = (desired != 0) & df[feat_columns].notna().all(axis=1)

                    prob = pd.Series(0.5, index=df.index)
                    if valid.any():
                        prob.loc[valid] = model.predict_proba(df.loc[valid, feat_columns])[
                            :, 1]  # вероятность, что сделка будет прибыльной

                    take_trade = (prob >= filt["prob_threshold"]).astype(
                        float)  # если вероятность выше порога, разрешаем сделку
                    confidence = (prob - filt["prob_threshold"]) / max(1e-9, 1.0 - filt[
                        "prob_threshold"])  # насколько уверенность модели выше минимального допустимого порога
                    confidence = np.clip(confidence, 0, 1)
                    position_size = filt["size_floor"] + (1.0 - filt[
                        "size_floor"]) * confidence  # чем выше уверенность модели, тем ближе размер позиции к полному
                    executed = desired * take_trade * position_size  # итоговая позиция

            asset_cap = self.cfg["asset_weight_cap"]  # максимум капитала
            beta_sign = np.sign(df["beta"].fillna(1.0)).replace(0, 1.0)

            exp_y_raw = executed * asset_cap
            exp_x_raw = -executed * beta_sign * asset_cap

            pair_exposures[key] = {
                "df": df,
                "exp_y_raw": exp_y_raw,
                "exp_x_raw": exp_x_raw
            }
        # хотим чтобы все было хорошо с загрузкой портфеля
        gross_raw = pd.Series(0.0, index=test_idx)
        for pack in pair_exposures.values():
            gross_raw = gross_raw.add(pack["exp_y_raw"].abs(), fill_value=0)
            gross_raw = gross_raw.add(pack["exp_x_raw"].abs(), fill_value=0)

        scale = pd.Series(1.0, index=test_idx)
        over = gross_raw > self.cfg["max_gross_exposure"]
        scale.loc[over] = self.cfg["max_gross_exposure"] / gross_raw.loc[over]

        portfolio_ret = pd.Series(0.0, index=test_idx)
        for pack in pair_exposures.values():
            df = pack["df"]
            exp_y = pack["exp_y_raw"] * scale
            exp_x = pack["exp_x_raw"] * scale

            gross_pair_ret = exp_y * df["o2o_y"].fillna(0.0) + exp_x * df["o2o_x"].fillna(0)  # доходность пары

            turnover_y = exp_y.diff().abs().fillna(exp_y.abs())  # учитываем транзакционные издержки
            turnover_x = exp_x.diff().abs().fillna(exp_x.abs())
            costs = (turnover_y + turnover_x) * self.cfg["transaction_cost"]

            net_pair_ret = gross_pair_ret - costs
            portfolio_ret = portfolio_ret.add(net_pair_ret,
                                              fill_value=0)  # складываем вклад всех пар в общую доходность портфеля

        self.daily_rets = portfolio_ret.fillna(0.0)
        self.equity = self.cfg["initial_capital"] * (1.0 + self.daily_rets).cumprod()
        return self.equity

    def report(self):  # считаем то что просили
        eq = self.equity
        rets = self.daily_rets.copy() if self.daily_rets is not None else eq.pct_change().fillna(0)

        total_ret = (eq.iloc[-1] - self.cfg["initial_capital"]) / self.cfg["initial_capital"]
        cumulative = (1 + rets).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        max_drawdown_period = 0
        if not cumulative.empty:
            peak_equity = cumulative.cummax()
            peak_indices = cumulative[cumulative == peak_equity].index

            if len(peak_indices) > 1:
                for i in range(len(peak_indices) - 1):
                    start_peak_date = peak_indices[i]
                    end_peak_date = peak_indices[i + 1]
                    duration = (end_peak_date - start_peak_date).days
                    if duration > max_drawdown_period:
                        max_drawdown_period = duration

        open_rets = (self.open_data.shift(-2) / self.open_data.shift(-1) - 1.0).loc[
            self.cfg["test_start"]:self.cfg["test_end"]].fillna(0)
        mkt = open_rets["BTC-USD"]
        var_mkt = mkt.var()

        portfolio_beta = rets.cov(mkt) / var_mkt if var_mkt else np.nan

        for p in self.pairs:
            y_name, x_name = p["y"], p["x"]
            beta_y = open_rets[y_name].cov(mkt) / var_mkt if var_mkt else 0
            beta_x = open_rets[x_name].cov(mkt) / var_mkt if var_mkt else 0
        sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() != 0 else np.nan
        years = len(eq) / 252
        cagr = (eq.iloc[-1] / self.cfg["initial_capital"]) ** (1 / years) - 1 if years > 0 else np.nan
        calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
        var_95 = np.percentile(rets, 5)
        es_95 = rets[rets <= var_95].mean()

        print(f"Капитал после торговли: ${eq.iloc[-1]:}")
        print(f"Доходность: {total_ret * 100:}%")
        print(f"Sharpe: {sharpe:.4f}")
        print(f"Максимальная просадка: {max_dd * 100:}%")
        print(f"Максимальная продолжительность просадки в днях: {max_drawdown_period}")
        print(f"Общая Бета портфеля к BTC-USD: {portfolio_beta:.4f}")
        print(f"Бета {y_name} к рынку: {beta_y:.4f}")
        print(f"Бета {x_name} к рынку: {beta_x:.4f}")
        print(f"Calmar: {calmar:}")
        print(f"VaR: {var_95 * 100:}%")
        print(f"ES: {es_95 * 100:}%")

        os.makedirs("artifacts", exist_ok=True)
        metrics = {
            "total_ret": float(total_ret),
            "sharpe": float(sharpe),
            "max_dd": float(max_dd),
            "portfolio_beta": float(portfolio_beta),
            "cagr": float(cagr),
            "calmar": float(calmar),
            "var_95": float(var_95),
            "es_95": float(es_95)
        }
        with open("artifacts/backtest_metrics.json", "w") as f:
            json.dump(metrics, f)

        plt.figure(figsize=(12, 6))
        plt.plot(eq)
        plt.title("Изменение капитала во времени")
        plt.grid(True)
        plt.legend()
        plt.savefig("artifacts/pnl_plot.png")
        plt.show()

        margin_requirement = self.equity * 0.1
        free_cash = self.equity - margin_requirement

        plt.figure(figsize=(12, 6))
        plt.fill_between(self.equity.index, 0, margin_requirement, label="Margin Requirements", color="red", alpha=0.3)
        plt.plot(free_cash, label="Free Cash Dynamics", color="green")
        plt.title("Requirements and Free Cash Dynamics")
        plt.legend()
        plt.grid(True)
        plt.savefig("artifacts/requirements_plot.png")
        plt.show()
