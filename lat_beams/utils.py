import os
import sys
import time

import numpy as np
import sqlalchemy as sqy
from astropy import units as u
from pixell import enmap, utils
from sotodlib.core import AxisManager, metadata
from sotodlib.preprocess.preprocess_util import preproc_or_load_group
from sotodlib.site_pipeline import jobdb
from sqlalchemy.pool import NullPool
from tqdm import tqdm
from sotodlib.mapmaking import init, ColoredFormatter
import logging
from logging.handlers import MemoryHandler

from .beam_utils import crop_maps

try:
    import mpi4py.rc

    mpi4py.rc.threads = False
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except:
    comm = None

def init_log(level=logging.DEBUG, comm=comm):
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
    L = init(level, rank = rank)
    for handler in L.handlers:
        if isinstance(handler.formatter, ColoredFormatter):
            handler.formatter.colors['NORMAL'] = "\033[1;34m"
    L.handlers = [MemoryHandler(1000, flushLevel=logging.CRITICAL, target=h, flushOnClose=True) for h in L.handlers]

    return L

def print_once(*args):
    """
    Helper function to print only once when running with MPI.
    Only the rank 0 process will print.

    Parameters
    ----------
    *args : Unpack[tuple[Any, ...]]
        Arguments to pass to print.
    """
    if comm is None or comm.Get_rank() == 0:
        print(*args)
        sys.stdout.flush()
class FakeJob:
    def __getattr__(self, name: str, /):
        _ = name
        return self._null_func

    def __setattr__(self, name: str, value):
        _ = name, value
        pass

    def _null_func(*args, **kwargs):
        _ = args, kwargs
        pass


def set_tag(job, key, new_val):
    if isinstance(job, FakeJob):
        return
    # This should be provided by the Job class but it's not...
    for _t in job._tags:
        if _t.key == key:
            _t.value = new_val
            return
    else:
        raise ValueError(f'No tag called "{key}"')


def load_aman(obs_id, preprocess_cfg, dets, job, min_dets, L, fp_flag=False):
    lvl = L.level
    try:
        L.setLevel(logging.ERROR)
        err, _, _, aman = preproc_or_load_group(
            obs_id,
            preprocess_cfg,
            dets=dets,
            save_archive=False,
            overwrite=True,
            logger=L,
        )
        L.setLevel(lvl)
    except:
        L.setLevel(lvl)
        msg = "Failed to load or preprocess!"
        L.error(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        return None
    if aman is None:
        msg = f"Preprocess failed with error {err}"
        L.error(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        return None

    if fp_flag:
        aman.restrict(
            "dets",
            np.isfinite(aman.focal_plane.xi)
            * np.isfinite(aman.focal_plane.eta)
            * np.isfinite(aman.focal_plane.gamma),
        )

    if aman.dets.count < min_dets:
        msg = f"Only {aman.dets.count} dets!"
        L.error(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        return None
    return aman


def make_jobdb(comm, data_dir):
    myrank = 0
    if comm is not None:
        myrank = comm.Get_rank()
    # Let rank 0 make jobdb first to avoid race conditions
    if myrank == 0:
        engine = sqy.create_engine(
            f'sqlite:///{os.path.join(data_dir, "jobdb.db")}',
            connect_args={"timeout": 10},
            poolclass=NullPool,
        )
        jdb = jobdb.JobManager(engine=engine)
        jdb.clear_locks(jobs="all")
        if comm is None:
            return jobdb
    comm.barrier()
    if myrank != 0:
        engine = sqy.create_engine(
            f'sqlite:///{os.path.join(data_dir, "jobdb.db")}',
            connect_args={"timeout": 10},
            poolclass=NullPool,
        )
        jdb = jobdb.JobManager(engine=engine)
    return jdb


def setup_jobs(
    comm,
    data_dir,
    jclass,
    get_jobdict,
    get_jobit,
    get_jobstr,
    get_tags,
    source_list,
    overwrite,
    retry_failed,
    job_memory,
    job_memory_buffer,
    replot,
    L,
):
    myrank = comm.Get_rank()
    nproc = comm.Get_size()
    # Get the jobs, make them if we need to
    now = time.time()
    L.info("Setting up jobdb")
    L.flush()
    jdb = make_jobdb(comm, data_dir)
    joblist = []
    jobs_to_make = []
    L.info("Getting jobdict")
    L.flush()
    jobdict = get_jobdict(jdb)
    L.info("Getting potential jobs")
    L.flush()
    it = get_jobit(jdb)
    L.info("Processing possible jobs")
    L.flush()
    for info in it:
        sys.stdout.flush()
        jobstr = get_jobstr(info)
        ignore_lock = False
        if jobstr is None:
            continue
        if jobstr in jobdict:
            job = jobdict[jobstr]
        else:
            tags = get_tags(info)
            job = jdb.create_job(jclass=jclass, tags=tags, check_existing=False, commit=False)
            jobs_to_make += [job]
            ignore_lock = True
        if job.lock and not ignore_lock:
            continue
        if job.tags["source"] not in source_list and job.tags["source"] != "":
            continue
        if (
            job.visit_time is not None
            and job_memory is not None
            and now - job.visit_time < 60 * 60 * job_memory
            and now - job.visit_time > 60 * job_memory_buffer
        ):
            continue
        if (
            overwrite
            or job.jstate.name == "open"
            or (job.jstate.name == "failed" and retry_failed)
        ):
            joblist += [job]
        elif replot and job.jstate.name == "done":
            joblist += [job]
    comm.barrier()

    # Make the missing jobs
    # Doing this serially so that we don't lock up the db
    tot_missing = 0
    tot_missing = comm.reduce(len(jobs_to_make), root=0)
    L.info(f"Adding {tot_missing} new jobs")
    L.flush()
    t0 = time.time()
    for i in range(nproc):
        if myrank == i:
            L.debug(f"\tRank {i} writing")
            jdb.commit_jobs(jobs_to_make)
            jdb.clear_locks(jobs=joblist)
        comm.barrier()
    t1 = time.time()
    L.flush()
    L.info(f"Took {t1-t0} seconds to add")

    # Get the final job list
    all_jobs = comm.allgather(joblist)
    all_jobs = [job for jobs in all_jobs for job in jobs]
    L.info(f"{len(all_jobs)} jobs to run!")
    L.flush()

    return jdb, all_jobs
