"""
Functions for fitting beam models to a map.

All functions will have the following standardized interface
which is defined by the `FitMap` protocol in this module.
Fitting functions should follow the naming convention `fit_{MODEL}_map`
where `{MODEL}` is a one word description of the model being fit.
"""

import logging
from typing import Optional, Protocol

from pixell.enmap import ndmap
from sotodlib.core import AxisManager
from sotodlib.tod_ops.filters import logger as flog

flog.setLevel(logging.ERROR)


class FitMap(Protocol):
    def __call__(
        self,
        imap: ndmap,
        ivar: ndmap,
        posmap: ndmap,
        guess: AxisManager,
        map_units: str = "pW",
        **kwargs,
    ) -> tuple[Optional[AxisManager], Optional[ndmap]]:
        """
        Function to fit a beam model to a map.

        Arguments
        ---------
        imap : ndmap
            Input map to fit with shape `(nx, ny)`.
        ivar : ndmap
            Inverse-variance map for `imap` with shape `(nx, ny)`.
        posmap : ndmap
            Position map in radians for `imap`.
            First element is eta and the second is xi.
            Should have shape `(2, nx, ny)`.
        guess : AxisManager
            `AxisManager` containing parameters that are useful as a starting point.
            See `make_guess` for the expected parameters.
        map_units : str, default: 'pW'
            The units of the map.
            Should be a string that astromy units understands.
        **kwargs
            Additional arguments for the specific fitting function.

        Returns
        -------
        fit_params : Optional[AxisManager]
            The fit parameters.
            See individual function docstrings for detail.
            Returns `None` if the fit failed.
        model : Optional[NDArray]
            The model evaluated with the fit parameters.
            Returns `None` if the fit failed.
        """
        pass


def make_guess(
    amp: float = 1,
    fwhm_xi: float = 2 / 60,
    fwhm_eta: float = 2 / 60,
    xi0: float = 0,
    eta0: float = 0,
    phi: float = 0,
    off: float = 0,
) -> AxisManager:
    """
    Helper function to make the initial guess `AxisManager`.
    Note that all arguments will be scalars in the output
    and all positional parameters are in radians.

    Arguments
    ---------
    amp : float, default: 1
        Amplitude of the beam.
    fwhm_xi : float, default: 2/60
        FWHM in xi.
    fwhm_eta : float, default: 2/60
        FWHM in eta.
    xi0 : float, default: 0
        Center of beam in xi.
    eta0 : float, default: 0
        Center of beam in eta.
    phi : float, default: 0
        Rotation of the beam.
    off : float, default: 0
        DC offset of the beam.

    Returns
    -------
    guess : AxisManager
        `AxisManager` with the guess parameters.
    """
    guess_dict = locals()
    guess = AxisManager()
    for n, v in guess_dict.items():
        guess.wrap(n, v)
    return guess
