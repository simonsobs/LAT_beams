import mpi4py

mpi4py.rc.threads = False
import sys

import numpy as np
from mpi4py import MPI

global_comm = MPI.COMM_WORLD
myrank = global_comm.Get_rank()
nproc = global_comm.Get_size()

if nproc == 1:
    raise ValueError("Too few procs! Run with at least two")

factor = 4

# Make a new comm for each group
mygroup = myrank % factor + factor * int(myrank == 0)
local_comm = global_comm.Split(mygroup, myrank)
stat = (
    myrank % factor + factor * int(myrank == 0),
    myrank,
    nproc,
    local_comm.Get_rank(),
    local_comm.Get_size(),
)
stat = global_comm.gather(stat, root=0)
if myrank == 0:
    stat = np.array(stat)
    stat = stat[np.argsort(stat[:, 0])]
    print(stat)

mytasks = local_comm.Get_size() * int(myrank > 0)
tot_tasks = global_comm.reduce(mytasks * int(local_comm.Get_rank() == 0))
if tot_tasks is None:
    tot_tasks = 0

global_comm.barrier()
# Do some task in the local groups
j = mygroup
for i in range(mytasks):
    local_comm.barrier()
    # Failure 1
    if i == j and j % 2 == 0:
        print("hi")
        if local_comm.Get_rank() == 0:
            global_comm.send(None, 0)
        continue

    s = 0
    s = local_comm.reduce(i + mygroup)

    # Failure 2
    if i == j and j % 2 == 0:
        if local_comm.Get_rank() == 0:
            global_comm.send(None, 0)
        continue

    # Have them all go to the global root
    if local_comm.Get_rank() == 0:
        global_comm.send(np.array([s]), 0)
# Receive them in the global root
for i in range(tot_tasks):
    s = global_comm.recv()
    if s is None:
        print(i, "failed")
        continue
    print(i, s)
