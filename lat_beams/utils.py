import os
import sys
import time

import numpy as np
import sqlalchemy as sqy
from astropy import units as u
from pixell import enmap, utils
from sotodlib.core import AxisManager
from sotodlib.preprocess.preprocess_util import preproc_or_load_group
from sotodlib.site_pipeline import jobdb
from sqlalchemy.pool import NullPool
from tqdm import tqdm

from .beam_utils import crop_maps

try:
    import mpi4py.rc

    mpi4py.rc.threads = False
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except:
    comm = None


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


def subpix_shift(imap, ishape, iwcs):
    crdelt = iwcs.wcs.crval - imap.wcs.wcs.crval
    cpdelt = iwcs.wcs.crpix - imap.wcs.wcs.crpix
    subpix = (crdelt / iwcs.wcs.cdelt - cpdelt + 0.5) % 1 - 0.5
    imap2 = enmap.fractional_shift(imap, -1 * subpix[::-1], nofft=False)

    return imap2


def coadd(imaps, iweights, medsub=True):
    # Get a joint geometry and init output maps
    ishape, iwcs = imaps[0].shape, imaps[0].wcs
    imaps = [subpix_shift(imap, ishape, iwcs) for imap in imaps]
    iweights = [subpix_shift(imap, ishape, iwcs) for imap in iweights]
    oshape, owcs = enmap.union_geometry([im.geometry for im in imaps])
    omap = enmap.zeros((len(imaps[0].shape),) + oshape, owcs)
    oweight = enmap.zeros((len(imaps[0].shape),) + oshape, owcs)

    # Coadd
    op = np.ndarray.__iadd__
    for im, iw in zip(imaps, iweights):
        oweight.insert(iw, op=op)
        off = 0
        # if medsub:
        #     off = utils.weighted_median(im, iw, axis=0)
        omap.insert(iw * (im - off), op=op)
    with np.errstate(divide="ignore", invalid="ignore"):
        omap /= oweight
    np.nan_to_num(omap, copy=False, nan=0, posinf=0, neginf=0)

    return omap, oweight


def recenter(imap, obs_id, stream_id, band, fit_file, norm=True, extent=None):
    aman_path = os.path.join(obs_id, stream_id, band)
    aman = AxisManager.load(fit_file, aman_path)
    cent = enmap.sky2pix(
        imap.shape,
        imap.wcs,
        np.array((aman.eta0.to(u.rad).value, aman.xi0.to(u.rad).value)),
    )
    zero = enmap.sky2pix(imap.shape, imap.wcs, np.array((0, 0)))
    imap = enmap.shift(imap, cent - zero)

    if norm:
        imap = (imap) / aman.amp.value

    if extent is not None:
        imap = crop_maps(
            [imap],
            enmap.sky2pix(imap.shape, imap.wcs, np.array((0, 0))),
            extent / np.abs(3600 * imap.wcs.wcs.cdelt[1]),
        )[0]

    return imap


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


def load_aman(obs_id, preprocess_cfg, dets, job, min_dets, fp_flag=False):
    try:
        err, _, _, aman = preproc_or_load_group(
            obs_id,
            preprocess_cfg,
            dets=dets,
            save_archive=False,
            overwrite=True,
        )
    except:
        msg = "Failed to load or preprocess!"
        print(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        return None
    if aman is None:
        msg = f"Preprocess failed with error {err}"
        print(f"\t{msg}")
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
        print(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        return None
    return aman


def make_jobdb(comm, data_dir):
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
):
    myrank = comm.Get_rank()
    nproc = comm.Get_size()
    # Get the jobs, make them if we need to
    now = time.time()
    print_once("Setting up jobdb")
    jdb = make_jobdb(comm, data_dir)
    joblist = []
    jobs_to_make = []
    print_once("Getting jobdict")
    jobdict = get_jobdict(jdb)
    print_once("Getting potential jobs")
    it = get_jobit(jdb)
    print_once("Processing possible jobs")
    if myrank == 0:
        it = tqdm(it, file=sys.stdout)
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
    print_once(f"Adding {tot_missing} new jobs")
    t0 = time.time()
    for i in range(nproc):
        print_once(f"\tRank {i} writing")
        if myrank == i:
            jdb.commit_jobs(jobs_to_make)
            jdb.clear_locks(jobs=joblist)
        comm.barrier()
    t1 = time.time()
    print_once(f"Took {t1-t0} seconds to add")

    # Get the final job list
    all_jobs = comm.allgather(joblist)
    all_jobs = [job for jobs in all_jobs for job in jobs]
    print_once(f"{len(all_jobs)} jobs to run!")

    return jdb, all_jobs
