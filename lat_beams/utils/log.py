"""
Utilities for managing logging.
"""

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

from pixell import colors
from sotodlib.mapmaking import ColoredFormatter, init

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

LoggerLike = logging.Logger | logging.LoggerAdapter[logging.Logger]


def init_log(
    level: int = logging.DEBUG, comm: Optional["Comm"] = comm
) -> logging.LoggerAdapter:
    """
    Initialize the sotodlib mapmaking logger with the following extra log levels:

    * A `NORMAL` log level (25) that is formatted as blue
    * A `DDEBUG` log level (5)

    This also uses `LoggerAdapter` to add a vairable called `extra`
    that is appended to the end of the the log message.
    It can be set with `logger.extra['extra'] = ...` and defaults to an
    empty string.

    Parameters
    ----------
    level : int, default: logging.DEBUG
        Logging level (e.g., logging.DEBUG, logging.INFO). Default is logging.DEBUG.
    comm : Optional[MPI.Comm], default: None
        An MPI communicator. If provided, the logger will include the rank in log messages.
        Default is None.

    Returns
    -------
    logging.LoggerAdapter
        The logger wrapped in a LoggerAdapter to add the `extra` formatting option.
    """

    def default_colfun(verbosity):
        cols = [
            colors.lpurple,
            colors.lred,
            "\033[1;34m",
            colors.lbrown,
            colors.lgreen,
            colors.reset,
        ]
        return cols[max(0, min(len(cols) - 1, verbosity + 3))]

    rank = 0
    if comm is not None:
        rank = comm.Get_rank()
    logging.addLevelName(25, "NORMAL")
    logging.addLevelName(5, "DDEBUG")
    fmt = "%(rank)3d %(wmins)7.2f %(resmem)5.2f %(mem)5.2f %(memmax)5.2f %(message)s%(extra)s"
    logger = init(level, rank=rank, fmt=fmt)
    for handler in logger.handlers:
        if isinstance(handler.formatter, ColoredFormatter):
            handler.formatter.colors = default_colfun
    logger = logging.LoggerAdapter(logger, {"extra": ""})

    return logger


@contextmanager
def log_lvl(
    logger: logging.Logger | logging.LoggerAdapter, level: Optional[int] = None
):
    """
    Temporarily set a logger (or LoggerAdapter) and its handlers to a different log level.
    Based on solution from StackOverflow: https://stackoverflow.com/q/78035371/850781

    Parameters
    ----------
    logger : logging.Logger | logging.LoggerAdapter
        The logger whose level should be temporarily changed.
    level : Optional[int], default: None
        The temporary log level to use. If None, increases the current effective
        level by 10. Default is None.

    Yields
    ------
    Tuple[int, List[int]]
        A tuple containing:
        * The original logger level.
        * A list of the original levels of all handlers.
    """
    if isinstance(logger, logging.LoggerAdapter):
        logger_use = logger.logger
    else:
        logger_use = logger
    saved_logger_level = logger_use.level
    saved_handler_levels = [ha.level for ha in logger_use.handlers]
    new_level = logger_use.getEffectiveLevel() + 10 if level is None else level
    logger_use.setLevel(new_level)
    for ha in logger_use.handlers:
        ha.setLevel(new_level)
    try:
        yield saved_logger_level, saved_handler_levels
    finally:
        logger_use.setLevel(saved_logger_level)
        for ha, le in zip(logger_use.handlers, saved_handler_levels):
            ha.setLevel(le)
