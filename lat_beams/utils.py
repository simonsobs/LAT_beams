import sys
import mpi4py.rc

mpi4py.rc.threads = False
from mpi4py import MPI

comm = MPI.COMM_WORLD


def print_once(*args):
    """
    Helper function to print only once when running with MPI.
    Only the rank 0 process will print.

    Parameters
    ----------
    *args : Unpack[tuple[Any, ...]]
        Arguments to pass to print.
    """
    if comm.Get_rank() == 0:
        print(*args)
        sys.stdout.flush()
