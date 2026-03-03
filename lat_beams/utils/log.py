import logging
import sys
from contextlib import contextmanager
from logging.handlers import MemoryHandler

from pshmem.locking import MPILock
from sotodlib.mapmaking import ColoredFormatter, init

try:
    import mpi4py.rc

    mpi4py.rc.threads = False
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except:
    comm = None


class MPIMemHandler(MemoryHandler):
    def createLock(self):
        """
        Acquire a thread lock for serializing access to the underlying I/O.
        """
        if comm:
            self.lock = MPILock(comm)

    def acquire(self):
        """
        Acquire the I/O thread lock.
        """
        if self.lock:
            self.lock.lock()

    def release(self):
        """
        Release the I/O thread lock.
        """
        if self.lock:
            self.lock.unlock()

    def handle(self, record):
        """
        Conditionally emit the specified logging record.

        Emission depends on filters which may have been added to the handler.
        Wrap the actual emission of the record with acquisition/release of
        the I/O thread lock.

        Returns an instance of the log record that was emitted
        if it passed all filters, otherwise a false value is returned.
        """
        rv = self.filter(record)
        if isinstance(rv, logging.LogRecord):
            record = rv
        if rv:
            self.emit(record)
        return rv

    def emit(self, record):
        """
        Emit a record.

        Append the record. If shouldFlush() tells us to, call flush() to process
        the buffer.
        """
        self.buffer.append(record)


def init_log(level=logging.DEBUG, comm=comm, flushLevel=logging.CRITICAL):
    # Uses a crappy version of https://stackoverflow.com/a/35804945
    def lognormal(self, message, *args, **kwargs):
        if self.isEnabledFor(25):
            self._log(25, message, args, **kwargs)

    def logddebug(self, message, *args, **kwargs):
        if self.isEnabledFor(5):
            self._log(5, message, args, **kwargs)

    def flush_log(self):
        for handler in self.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

    rank = 0
    if comm is not None:
        rank = comm.Get_rank()
    logging.addLevelName(25, "NORMAL")
    logging.addLevelName(5, "DDEBUG")
    setattr(logging.getLoggerClass(), "normal", lognormal)
    setattr(logging.getLoggerClass(), "ddebug", logddebug)
    setattr(logging.getLoggerClass(), "flush", flush_log)
    logger = init(level, rank=rank)
    for handler in logger.handlers:
        if isinstance(handler.formatter, ColoredFormatter):
            handler.formatter.colors["NORMAL"] = "\033[1;34m"
    logger.handlers = [
        MPIMemHandler(1000, flushLevel=flushLevel, target=h, flushOnClose=True)
        for h in logger.handlers
    ]

    return logger


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
