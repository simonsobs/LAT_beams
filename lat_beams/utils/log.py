import logging
from contextlib import contextmanager
from logging.handlers import MemoryHandler

from sotodlib.mapmaking import ColoredFormatter, init

try:
    import mpi4py.rc

    mpi4py.rc.threads = False
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except:
    comm = None


def init_log(level=logging.DEBUG, comm=comm, flushLevel=logging.CRITICAL):
    # Uses a crappy version of https://stackoverflow.com/a/35804945
    def lognormal(self, message, *args, **kwargs):
        if self.isEnabledFor(25):
            self._log(25, message, args, **kwargs)

    def flush_log(self):
        for handler in self.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

    rank = 0
    if comm is not None:
        rank = comm.Get_rank()
    logging.addLevelName(25, "NORMAL")
    setattr(logging.getLoggerClass(), "normal", lognormal)
    setattr(logging.getLoggerClass(), "flush", flush_log)
    L = init(level, rank=rank)
    for handler in L.handlers:
        if isinstance(handler.formatter, ColoredFormatter):
            handler.formatter.colors["NORMAL"] = "\033[1;34m"
    L.handlers = [
        MemoryHandler(1000, flushLevel=flushLevel, target=h, flushOnClose=True)
        for h in L.handlers
    ]

    return L


@contextmanager
def log_lvl(logger, level=None):
    "Run body with logger at a different level."
    # https://stackoverflow.com/q/78035371/850781
    saved_logger_level = logger.level
    saved_handler_levels = [ha.level for ha in logger.handlers]
    new_level = logger.getEffectiveLevel() + 10 if level is None else level
    logger.setLevel(new_level)
    for ha in logger.handlers:
        ha.setLevel(new_level)
    try:
        yield saved_logger_level, saved_handler_levels
    finally:
        logger.setLevel(saved_logger_level)
        for ha, le in zip(logger.handlers, saved_handler_levels):
            ha.setLevel(le)
