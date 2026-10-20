"""
Tools for setting up the JobDB.

TODO: Add some quick tools for inspecting the jobdb here.
"""

import os
import shutil
import sqlite3
import sys
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence, cast

import sqlalchemy as sqy
from sotodlib.site_pipeline import jobdb
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from .log import LoggerLike

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


class ErrCode(Enum):
    NO_ERR = 0
    META = 1
    PREPROC = 2
    MIN_DETS = 3
    SRC_TAG = 4
    PLOT = 5
    ML_MAP = 6
    MAP_MISSING = 7
    ZERO_IVAR = 8
    SNR_LOW = 9
    FIT_FAILED = 10
    CLOSE_TO_EDGE = 11
    FWHM_TOL = 12
    DET_SECS = 13
    NO_MAPS = 14
    NO_JOB = 15
    FILT_FAILED = 16
    MAP_FAILED = 17
    OMEGA_FAILED = 18
    FIT_MISSING = 19


def set_tag(job, key, new_val):
    # This should be provided by the Job class but it's not...
    for _t in job._tags:
        if _t.key == key:
            _t.value = new_val
            return
    else:
        raise ValueError(f'No tag called "{key}"')


def fail(job: jobdb.Job, errcode: ErrCode, msg: str, logger: Optional[LoggerLike]):
    """
    Mark and job as failed.

    Parameters
    ----------
    job : jobdb.Job
        The job to fail.
    errcode : ErrCode
        The error code that thi job failed with.
    msg : str
        The detailed error message.
    logger : LoggerLike
        The logger to log the error to.
    """
    if logger is not None:
        logger.error("%s (Err %d: %s)", msg, errcode.value, errcode.name)
    set_tag(job, "message", msg)
    if "errcode" in job.tags:
        set_tag(job, "errcode", errcode.value)
    job.jstate = cast(sqy.Column[str], jobdb.JState.failed)


def _sync_jobs(db_a_path, db_b_path):
    conn_a = sqlite3.connect(db_a_path)
    conn_b = sqlite3.connect(db_b_path)

    try:
        jobs = conn_a.execute("""
            SELECT
                id,
                jclass,
                jstate,
                lock,
                lock_owner,
                creation_time,
                visit_time,
                visit_count
            FROM jobs
            WHERE jclass IN ('beam_map', 'fit_map')
        """).fetchall()

        for job in jobs:
            (
                job_id,
                jclass,
                jstate,
                lock,
                lock_owner,
                creation_time,
                visit_time,
                visit_count,
            ) = job

            # Look for the same job in B.
            existing = conn_b.execute(
                """
                SELECT visit_time
                FROM jobs
                WHERE id = ?
            """,
                (job_id,),
            ).fetchone()

            # Copy if the job doesn't exist in B.
            if existing is None:
                should_copy = True
            else:
                b_visit_time = existing[0]

                # Copy only when visit_time is exactly the same.
                should_copy = visit_time == b_visit_time

            if not should_copy:
                continue

            # Insert/replace the job.
            conn_b.execute(
                """
                INSERT OR REPLACE INTO jobs (
                    id,
                    jclass,
                    jstate,
                    lock,
                    lock_owner,
                    creation_time,
                    visit_time,
                    visit_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                job,
            )

            conn_b.execute(
                """
                DELETE FROM tags
                WHERE job_id = ?
            """,
                (job_id,),
            )

            tags = conn_a.execute(
                """
                SELECT "key", value
                FROM tags
                WHERE job_id = ?
            """,
                (job_id,),
            ).fetchall()

            conn_b.executemany(
                """
                INSERT INTO tags (job_id, "key", value)
                VALUES (?, ?, ?)
            """,
                [(job_id, key, value) for key, value in tags],
            )
        conn_b.commit()
    except Exception:
        conn_b.rollback()
        raise
    finally:
        conn_a.close()
        conn_b.close()


def make_jobdb(
    comm: Optional["Comm"], data_dir: str, append: str = ""
) -> jobdb.JobManager:
    """
    Create or load a `JobDB` at `{data_dir}/jobdb{append}.db`.
    If a jobdb without `append` exists its entries will be merged
    into the `append` jobdb based on the last update.

    Note that this can create some confusion because it may reference
    jobs that were run in the main jobdb, to address this jobs with jclass
    `stack_maps` and `fit_stacks` are not copied, but use with caution
    if you are running jobs of jclass `source_map` or `fit_map`.
    If you are running one of those some manual reconfiguring may be needed.
    This should be used to test new methods or create custom one-off stacks.

    Parameters
    ----------
    comm : Optional[Comm]
        The communicator if we are running with MPI or None if not.
        If provided then rank 0 will load the database first so that it may
        create it if it doesn't exist.
    data_dir : str
        The directory to load the db from.
    append : str
        String appended to db name.
        See docstring for details.

    Returns
    -------
    jobdb : jobdb.JobManager
        The loaded database.
        For better MPI support this has a timeout of 10 and uses NullPool.
    """
    path = os.path.join(data_dir, f"jobdb{append}.db")
    path_noa = os.path.join(data_dir, f"jobdb.db")
    myrank = 0
    if comm is not None:
        myrank = comm.Get_rank()
    # Let rank 0 make jobdb first to avoid race conditions
    jdb = None
    if myrank == 0:
        if append != "" and os.path.isfile(path_noa):
            if os.path.isfile(path):
                _sync_jobs(path_noa, path)
            else:
                shutil.copyfile(path_noa, path)
        engine = sqy.create_engine(
            f"sqlite:///{path}",
            connect_args={"timeout": 10},
            poolclass=NullPool,
        )
        jdb = jobdb.JobManager(engine=engine)
        jdb.clear_locks(jobs="all")

        if comm is None:
            return jdb
    if comm is not None:
        comm.barrier()
    if myrank != 0:
        engine = sqy.create_engine(
            f"sqlite:///{path}",
            connect_args={"timeout": 10},
            poolclass=NullPool,
        )
        jdb = jobdb.JobManager(engine=engine)
    if jdb is None:
        raise ValueError("Jobdb is none somehow!")
    return jdb


def setup_jobs(
    comm: Optional["Comm"],
    data_dir: str,
    jclass: str,
    get_jobdict: Callable[[jobdb.JobManager], dict[str, jobdb.Job]],
    get_jobit: Callable[[jobdb.JobManager], Iterable[Any]],
    get_jobstr: Callable[[Any], Optional[str]],
    get_tags: Callable[[Any], dict[str, str]],
    source_list: Sequence[str],
    overwrite: bool,
    retry_failed: bool,
    job_memory: Optional[float],
    job_memory_buffer: float,
    replot: bool,
    logger: LoggerLike,
    append: str = "",
) -> tuple[jobdb.JobManager, list[jobdb.Job]]:
    """
    Discover, create, and select jobs for execution across MPI ranks.

    Existing jobs are loaded from the job database and matched against a
    collection of candidate jobs generated by `get_jobit`. Missing jobs are
    created, eligible jobs are reopened when requested, and a consolidated
    list of jobs to process is returned.

    Jobs may be filtered by lock status, source tag, recent visit time, and
    job state. Database writes are serialized across MPI ranks to avoid
    contention.

    Jobs are created and database updates are committed serially across MPI
    ranks to avoid database locking contention.

    Parameters
    ----------
    comm : Optional[Comm]
        MPI communicator used to prevent deadlock between processes.
    data_dir : str
        Directory containing the job database.
    jclass : str
        Job class name passed to `JobManager.create_job` when creating
        missing jobs.
    get_jobdict : Callable[[jobdb.JobManager], dict[str, jobdb.Job]]
        Function that returns a mapping from job identifier strings to
        existing jobs in the database.
    get_jobit : Callable[[jobdb.JobManager], Iterable[Any]]
        Function that returns an iterable of candidate job descriptions.
    get_jobstr : Callable[[Any], Optional[str]]
        Function that converts a candidate job description into its unique
        job identifier string. Returning `None` causes the candidate to be
        skipped.
    get_tags : Callable[[Any], dict[str, str]]
        Function that generates the tag dictionary used when creating a new
        job from a candidate job description.
    source_list : Sequence[str]
        Allowed values of the `source` tag. Jobs whose source tag is not
        present in this sequence are ignored.
    overwrite : bool
        If `True`, include all matching jobs regardless of their current
        state and reopen non-open jobs.
    retry_failed : bool
        If `True`, include jobs whose state is `failed`.
    job_memory : Optional[float]
        Number of hours for which recently visited jobs should be skipped.
        If `None`, no visit-time filtering is performed.
    job_memory_buffer : float
        Minimum age, in minutes, before the visit-time filter is applied.
    replot : bool
        If `True`, include jobs whose state is `done`.
    logger : LoggerLike
        Logger to log to.
    append : str
        String appended to db name.
        See `make_jobdb` docstring for details.

    Returns
    -------
    jdb : jobdb.JobManager
        Job database manager instance.
    jobs : list[jobdb.Job]
        Complete list of jobs selected for processing across all MPI ranks.
    """
    if append != "" and jclass in ["beam_map", "fit_map"]:
        logger.warning(
            "Appending %s to jobdb but running %s jobs, this can me messy, make sure you know what you are doing",
            append,
            jclass,
        )
    myrank, nproc = 0, 1
    if comm is not None:
        myrank = comm.Get_rank()
        nproc = comm.Get_size()
    # Get the jobs, make them if we need to
    now = time.time()
    logger.info("Setting up jobdb")
    jdb = make_jobdb(comm, data_dir, append)
    joblist = []
    jobs_to_make = []
    jobs_to_open = []
    logger.info("Getting jobdict")
    jobdict = None
    if myrank == 0:
        jobdict = get_jobdict(jdb)
    if comm is not None:
        jobdict = comm.bcast(jobdict)
    if jobdict is None:
        raise ValueError("jobdict is None!")
    logger.info("Getting potential jobs")
    it = get_jobit(jdb)
    logger.info("Processing possible jobs")
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
        if (
            "source" in job.tags
            and job.tags["source"] not in source_list
            and job.tags["source"] != ""
        ):
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
            if job.jstate.name != "open":
                job.jstate = "open"
                jobs_to_open += [job]
                # joblist += [job]
            else:
                joblist += [job]
        elif replot and job.jstate.name == "done":
            joblist += [job]
    if comm is not None:
        comm.barrier()

    # Make the missing jobs
    # Doing this serially so that we don't lock up the db
    tot_missing = len(jobs_to_make)
    if comm is not None:
        tot_missing = comm.reduce(len(jobs_to_make), root=0)
    logger.info("Adding %s new jobs", tot_missing)
    tot_opening = len(jobs_to_open)
    if comm is not None:
        tot_opening = comm.reduce(len(jobs_to_open), root=0)
    logger.info("Opening %s old jobs", tot_opening)
    t0 = time.time()
    for i in range(nproc):
        if myrank == i:
            logger.debug("\tRank %s writing", i)
            jdb.commit_jobs(jobs_to_make)
            jdb.clear_locks(jobs=joblist)
            if len(jobs_to_open) > 0:
                jdb.update_jobs(jobs_to_open)
                with jdb.session_scope() as session:
                    for job in jobs_to_open:  # updated_jobs:
                        jid = job.id
                        refreshed_job = session.get(jobdb.Job, jid)
                        session.expunge(refreshed_job)
                        joblist.append(refreshed_job)
        if comm is not None:
            comm.barrier()
    t1 = time.time()
    logger.info("Took %s seconds to add", t1 - t0)

    # Get the final job list
    if comm is not None:
        all_jobs = comm.allgather(joblist)
        all_jobs = [job for jobs in all_jobs for job in jobs]
    else:
        all_jobs = joblist
    logger.info("%s jobs to run!", len(all_jobs))

    return jdb, all_jobs


def update_jobs_retry(
    jdb: jobdb.JobManager,
    jobs: Sequence[jobdb.Job],
    max_retries: int,
    logger: LoggerLike,
):
    """
    Update jobs in the job database, retrying on database lock errors.

    The update is retried up to `max_retries` times when the database
    reports that it is locked. A one-second delay is inserted between
    retries. Other operational errors are propagated immediately.

    Parameters
    ----------
    jdb : jobdb.JobManager
        Job database manager used to update the jobs.
    jobs : Sequence[jobdb.Job]
        Jobs to update in the database.
    max_retries : int
        Maximum number of attempts to make when updating the jobs.
    logger : LoggerLike
        Logger used to report failures and successful updates.

    Raises
    ------
    OperationalError
        If an operational database error occurs that is not caused by
        the database being locked.
    """
    t0 = time.time()
    attempt = 0
    success = False

    for attempt in range(max(1, max_retries)):
        try:
            jdb.update_jobs(jobs)
            success = True
            break
        except OperationalError as e:
            if "database is locked" in str(e):
                time.sleep(1)
                continue
            raise

    if not success:
        logger.error("Failed to write with %d attempts", attempt + 1)
    else:
        logger.debug(
            "Took %s seconds to write with %d attempts",
            str(time.time() - t0),
            attempt + 1,
        )
