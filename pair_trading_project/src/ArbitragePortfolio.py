from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PairTradingBacktest:
    """
    Absolutely ineffective and non-user-friendly pair trading backtest class ◝(ᵔᗜᵔ)◜
    """

    def __init__(
        self,
        close,
        units_held,
        pnl,
        free_cash,
        quasi_cash,
        fees,
        interests,
        margins_required,
    ):
        self.close = close
        self.units_held = units_held
        self.pnl = pnl
        self.free_cash = free_cash
        self.quasi_cash = quasi_cash
        self.fees = fees
        self.interests = interests
        self.margins_required = margins_required

    @staticmethod
    def __rebalance_pf(
        close,
        shares_a_before,
        shares_a,
        shares_b_before,
        shares_b,
        free_cash,
        quasi_cash,
        fees,
    ):
        commissions = (
            abs(shares_a - shares_a_before) * close.iloc[0]
            + abs(shares_b - shares_b_before) * close.iloc[1]
        ) * fees

        short_position = (
            max(0, -shares_a) * close.iloc[0] + max(0, -shares_b) * close.iloc[1]
        )

        long_position = (
            max(0, shares_a) * close.iloc[0] + max(0, shares_b) * close.iloc[1]
        )

        free_cash = (
            free_cash
            - commissions
            + (max(shares_a_before, 0) - max(shares_a, 0)) * close.iloc[0]
            + (max(shares_b_before, 0) - max(shares_b, 0)) * close.iloc[1]
        )

        quasi_cash = (
            quasi_cash
            + (max(-shares_a, 0) - max(-shares_a_before, 0)) * close.iloc[0]
            + (max(-shares_b, 0) - max(-shares_b_before, 0)) * close.iloc[1]
        )

        value = free_cash + quasi_cash + long_position - short_position
        return free_cash, quasi_cash, long_position, short_position, value, commissions

    @classmethod
    def backtest(
        cls,
        close: pd.DataFrame,
        size: pd.DataFrame,
        init_cash: float = 1,
        fees: float = 0,
        slippage: float = 0,
        interest_rate: float = 0.0001,
        rebate_rate: float = 0.00003,
        initial_margin_rate: float = 0.5,
        maintenance_margin_rate: float = 0.3,
        collateral: float = 1.02,
        margin_handle="partial_closing",  # partial_closing, forced_closing, attract_debt, raise_error
    ):
        """Backtesting pair trading strategy. Analogue of vectorbt.from_orders.

        Args:
            close (pd.DataFrame): Close price
            size (pd.DataFrame): VectorBT target_percent dataframe
            init_cash (float, optional): Initial capital
            fees (float, optional): Variable fees rate. Defaults to 0.0.
            slippage (float, optional): Slippage percent. Defaults to 0.0.
            interest_rate (float, optional): Loan fee rate (not adjusted for strategy timeframe). Defaults to 0.0001.
            rebate_rate (float, optional): Rebate rate (not adjusted for strategy timeframe). Defaults to 0.00003.
            initial_margin_rate (float, optional): Initial margin requirement. Set by Regulation T. Defaults to 0.5.
            maintenance_margin_rate (float, optional): Maintenance margin requirement. Set by broker. Defaults to 0.3.
            collateral (float, optional): usually 102% in US and 105% for international securities. Defaults to 1.02.
            margin_handle (str, optional):
                - partial_closing: Partially close the position to relief necessary cash to cover margin call.
                - forced_closing: Close entire position.
                - attract_debt: Leverage the position.
                - raise_error: raise error
                Defaults to partial_closing.

        Raises:
            Warning: Margin Call

        Returns:
            PairTradingBacktes: Initialized portfolio object
        """
        pnl = pd.Series(index=close.index)
        units_held = pd.DataFrame(index=close.index, columns=close.columns)
        free_cash_series = pd.Series(index=close.index)
        quasi_cash_series = pd.Series(index=close.index)
        fees_series = pd.Series(index=close.index)
        interest_series = pd.Series(index=close.index)
        margin_required_series = pd.Series(index=close.index)

        value = init_cash  # net cash value of our portfolio = free_cash + quasi_cash + value of long position - value of short position
        free_cash = init_cash
        quasi_cash = 0  # sell price * stocks shorted
        shares_a = shares_a_before = 0
        shares_b = shares_b_before = 0
        short_position = 0
        long_position = 0

        for idx in close.index:
            try:
                free_cash -= (interest_rate - rebate_rate) * short_position * collateral
                shares_a_before = shares_a
                shares_a = (
                    size.loc[idx, :].iloc[0]
                    * value
                    / (1 + fees + slippage)
                    / close.loc[idx, :].iloc[0]
                    if size.loc[idx, :].iloc[0] is not np.nan
                    else shares_a
                )

                shares_b_before = shares_b
                shares_b = (
                    size.loc[idx, :].iloc[1]
                    * value
                    / (1 + fees + slippage)
                    / close.loc[idx, :].iloc[1]
                    if size.loc[idx, :].iloc[1] is not np.nan
                    else shares_b
                )

                (
                    free_cash,
                    quasi_cash,
                    long_position,
                    short_position,
                    value,
                    commisions,
                ) = cls.__rebalance_pf(
                    close.loc[idx, :],
                    shares_a_before,
                    shares_a,
                    shares_b_before,
                    shares_b,
                    free_cash,
                    quasi_cash,
                    fees,
                )

            except KeyError:
                pass

            # Margin Call
            if size.loc[idx, :].iloc[0] is not np.nan:  # if today is rebalance day
                margin_required = short_position * initial_margin_rate * collateral
            else:
                sell_price = (
                    quasi_cash / (max(0, -shares_a) + max(0, -shares_b))
                    if quasi_cash != 0
                    else 0
                )

                current_price = (
                    close.loc[idx, :].iloc[0]
                    if shares_a < 0
                    else close.loc[idx, :].iloc[1]
                )
                if sell_price > current_price:
                    margin_required = (
                        short_position * (1 + maintenance_margin_rate) * collateral
                        - short_position
                    )
                else:
                    margin_required = (
                        short_position * (initial_margin_rate + 1) * collateral
                        - short_position
                    )

            additional_margin_requirements = margin_required - free_cash

            if additional_margin_requirements > 0:
                if margin_handle == "partial_closing":
                    print(
                        f"Warning: Margin Call at {datetime.strftime(idx, '%Y-%m-%d')}"
                    )

                    shares_a_before = shares_a
                    shares_a = shares_a * (
                        1
                        - additional_margin_requirements
                        / (value * (1 + fees + slippage))
                    )

                    shares_b_before = shares_b
                    shares_b = shares_b * (
                        1
                        - additional_margin_requirements
                        / (value * (1 + fees + slippage))
                    )

                    (
                        free_cash,
                        quasi_cash,
                        long_position,
                        short_position,
                        value,
                        commisions,
                    ) = cls.__rebalance_pf(
                        close.loc[idx, :],
                        shares_a_before,
                        shares_a,
                        shares_b_before,
                        shares_b,
                        free_cash,
                        quasi_cash,
                        fees,
                    )

                if margin_handle == "forced_closing":
                    shares_a_before = shares_a
                    shares_a = 0

                    shares_b_before = shares_b
                    shares_b = 0
                    (
                        free_cash,
                        quasi_cash,
                        long_position,
                        short_position,
                        value,
                        commisions,
                    ) = cls.__rebalance_pf(
                        close.loc[idx, :],
                        shares_a_before,
                        shares_a,
                        shares_b_before,
                        shares_b,
                        free_cash,
                        quasi_cash,
                        fees,
                    )

                if margin_handle == "raise_error":
                    raise Warning("Warning: Margin Call! Cancel sumulation")
                    return

            pnl[idx] = value
            free_cash_series[idx] = free_cash
            quasi_cash_series[idx] = quasi_cash
            units_held.loc[idx, :] = shares_a, shares_b
            fees_series[idx] = commisions
            interest_series[idx] = (
                (interest_rate - rebate_rate) * short_position * collateral
            )
            margin_required_series[idx] = margin_required

        return cls(
            close,
            units_held,
            pnl,
            free_cash_series,
            quasi_cash_series,
            fees_series,
            interest_series,
            margin_required_series,
        )

    def plot_pnl(self, output_file=None):
        """Plotting PNL, assets prices and positions dynamics"""
        fig, ax = plt.subplots(
            3, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [2, 0.75, 2]}
        )

        ax[0].plot(self.close.iloc[:, 0] / self.close.iloc[0, 0], c="0", linewidth=1)
        ax[0].plot(self.close.iloc[:, 1] / self.close.iloc[0, 1], c="r", linewidth=1)
        ax[0].title.set_text("Normalized Prices")
        ax[0].grid()
        ax[0].legend(self.close.columns)

        ax[1].plot(self.units_held.iloc[:, 0], c="0")
        ax[1].plot(self.units_held.iloc[:, 1], c="r")
        ax[1].legend(self.units_held.columns)
        ax[1].title.set_text("Units Held")

        ax[2].plot(self.pnl, c="g", linewidth=1)
        ax[2].plot(
            self.pnl + self.fees.fillna(0).cumsum() + self.interests.fillna(0).cumsum(),
            c="0",
            linewidth=1,
        )
        ax[2].title.set_text("Equity Curve: P&L")
        ax[2].grid()
        ax[2].legend(["Net value", "Gross value"])

        if output_file is not None:
            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        else:
            plt.show()

    def plot_requirements(self, output_file=None):
        """Plotting margin requirements and free cash"""
        fig, ax = plt.subplots(
            2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 1]}
        )

        ax[0].plot(
            (self.close * self.units_held.abs()).sum(axis=1) / self.pnl,
            c="0",
        )
        ax[0].plot(
            (
                self.close
                * (self.units_held.clip(lower=0) + self.units_held.clip(upper=0))
            ).sum(axis=1)
            / self.pnl,
            c="r",
        )

        ax[0].legend(["Gross exposure", "Net exposure"])
        ax[0].title.set_text("Net/Gross exposure")

        ax[1].plot(self.free_cash, c="0")
        ax[1].plot(
            self.margins_required,
            c="r",
        )
        ax[1].legend(["Free cash", "Margin requirements"])
        ax[1].title.set_text("Short position service")
        ax[1].grid()

        if output_file is not None:
            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        else:
            plt.show()

    def stats(self, freq="D"):
        """Quick backtesting metrics

        Args:
            freq (str, optional): Frequency of your dataframe. Defaults to "D".

        Returns:
            None
        """
        scale_factor = {"D": np.sqrt(252), "M": np.sqrt(12), "Y": 1}

        total_return = round(self.pnl.iloc[-1] / self.pnl.iloc[0] - 1, 4)
        information_ratio = (
            scale_factor.get(freq)
            * self.pnl.pct_change().mean()
            / self.pnl.pct_change().std()
        )
        max_drawdown = (self.pnl / self.pnl.cummax() - 1).min()
        volatility = scale_factor.get(freq) * self.pnl.pct_change().std()
        var = self.pnl.pct_change().quantile(0.05)

        stats_dict = {}
        stats_dict["net_return_pct"] = round(100 * total_return, 2)
        stats_dict["information_ratio"] = round(information_ratio, 2)
        stats_dict["max_drawdown_pct"] = round(100 * max_drawdown, 2)
        stats_dict["annual_volatility_pct"] = round(100 * volatility, 2)
        stats_dict["var_5_pct"] = round(100 * var, 2)

        return stats_dict
