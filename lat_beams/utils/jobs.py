import os
import sys
import time

import sqlalchemy as sqy
from sotodlib.site_pipeline import jobdb
from sqlalchemy.pool import NullPool


def set_tag(job, key, new_val):
    # This should be provided by the Job class but it's not...
    for _t in job._tags:
        if _t.key == key:
            _t.value = new_val
            return
    else:
        raise ValueError(f'No tag called "{key}"')


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
            return jdb
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
    jobdict = None
    if myrank == 0:
        jobdict = get_jobdict(jdb)
    jobdict = comm.bcast(jobdict)
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
            job = jdb.create_job(
                jclass=jclass, tags=tags, check_existing=False, commit=False
            )
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
