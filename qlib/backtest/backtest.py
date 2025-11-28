# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, TYPE_CHECKING, Generator, Optional, Tuple, Union, cast

import pandas as pd

from qlib.backtest.decision import BaseTradeDecision
from qlib.backtest.report import Indicator

if TYPE_CHECKING:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.executor import BaseExecutor

from tqdm.auto import tqdm

from ..utils.time import Freq


PORT_METRIC = Dict[str, Tuple[pd.DataFrame, dict]]
INDICATOR_METRIC = Dict[str, Tuple[pd.DataFrame, Indicator]]


def backtest_loop(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    trade_strategy: BaseStrategy,
    trade_executor: BaseExecutor,
) -> Tuple[PORT_METRIC, INDICATOR_METRIC]:
    """backtest function for the interaction of the outermost strategy and executor in the nested decision execution

    please refer to the docs of `collect_data_loop`

    Returns
    -------
    portfolio_dict: PORT_METRIC
        it records the trading portfolio_metrics information
    indicator_dict: INDICATOR_METRIC
        it computes the trading indicator
    """
    return_value: dict = {}
    for _decision in collect_data_loop(start_time, end_time, trade_strategy, trade_executor, return_value):
        pass

    portfolio_dict = cast(PORT_METRIC, return_value.get("portfolio_dict"))
    indicator_dict = cast(INDICATOR_METRIC, return_value.get("indicator_dict"))

    return portfolio_dict, indicator_dict


def collect_data_loop(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    trade_strategy: BaseStrategy,
    trade_executor: BaseExecutor,
    return_value: dict | None = None,
) -> Generator[BaseTradeDecision, Optional[BaseTradeDecision], None]:
    """Generator for collecting the trade decision data for rl training

    通过 yeild from 返回一个 decision 对象, 最后将结果写入 return_value，
    上层调用者从传入的 return_value 中获取结果

    Parameters
    ----------
    start_time : Union[pd.Timestamp, str]
        closed start time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
    end_time : Union[pd.Timestamp, str]
        closed end time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
        E.g. Executor[day](Executor[1min]), setting `end_time == 20XX0301` will include all the minutes on 20XX0301
    trade_strategy : BaseStrategy
        the outermost portfolio strategy
    trade_executor : BaseExecutor
        the outermost executor
    return_value : dict
        used for backtest_loop

        mc: python 在函数传入的类型为list (列表)、dict (字典)、set (集合)、用户自定义的类实例时，
        为可变对象，函数内对其进行修改会影响到函数外的值。
        因此通过传入一个空字典，在函数内修改这个字典，函数外也能获取到修改后的值。

        return_value 中有两个 key:
        - "portfolio_dict": 记录各个 executor 的 portfolio_metrics 信息
        - "indicator_dict": 记录各个 executor 的 trade_indicator 信息

    Yields
    -------
    object
        trade decision
    """
    # 在 backtest_loop 中会被调用，这两句只执行一次
    # executer.reset() 会调用到 LevelInfrastructure::reset_cal() 该函数会创建 trade_calendar
    # 
    trade_executor.reset(start_time=start_time, end_time=end_time)
    trade_strategy.reset(level_infra=trade_executor.get_level_infra())


    with tqdm(total=trade_executor.trade_calendar.get_trade_len(), desc="backtest loop") as bar:
        _execute_result = None
        while not trade_executor.finished():
            _trade_decision: BaseTradeDecision = trade_strategy.generate_trade_decision(_execute_result)

            # mc: 在 collect_data 调用 trade_calendar.step() 来推进时间
            # yeild from 语法将迭代委托给 collect_data 方法
            # 如果 collect_data 方法没有 yield 语句，则相当于一个普通的函数调用
            _execute_result = yield from trade_executor.collect_data(_trade_decision, level=0)

            trade_strategy.post_exe_step(_execute_result)

            # 显示 
            bar.update(1)
        trade_strategy.post_upper_level_exe_step()

    if return_value is not None:
        all_executors = trade_executor.get_all_executors()

        portfolio_dict: PORT_METRIC = {}
        indicator_dict: INDICATOR_METRIC = {}

        for executor in all_executors:
            key = "{}{}".format(*Freq.parse(executor.time_per_step))
            if executor.trade_account.is_port_metr_enabled():
                portfolio_dict[key] = executor.trade_account.get_portfolio_metrics()

            indicator_df = executor.trade_account.get_trade_indicator().generate_trade_indicators_dataframe()
            indicator_obj = executor.trade_account.get_trade_indicator()
            indicator_dict[key] = (indicator_df, indicator_obj)

        return_value.update({"portfolio_dict": portfolio_dict, "indicator_dict": indicator_dict})

