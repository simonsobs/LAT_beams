"""
Interfaces and helpers for fitting beam models to maps.

Beam-fitting functions follow the `fit_{MODEL}_map` naming convention and
the `FitMap` interface defined below. See individual fitting functions
for details of each beam model.
"""

import logging
from typing import Optional, Protocol

from jaxtyping import Float
from pixell.enmap import ndmap
from sotodlib.core import AxisManager
from sotodlib.tod_ops.filters import logger as flog

flog.setLevel(logging.ERROR)


class FitMap(Protocol):
    def __call__(
        self,
        imap: Float[ndmap, "ny nx"],
        ivar: Float[ndmap, "ny nx"],
        posmap: Float[ndmap, "2 ny nx"],
        guess: AxisManager,
        map_units: str = "pW",
        **kwargs: object,
    ) -> tuple[Optional[AxisManager], Optional[Float[ndmap, "ny nx"]]]:
        """Fit a beam model to a map.

        Parameters
        ----------
        imap : Float[ndmap, "ny nx"]
            Input beam map.
        ivar : Float[ndmap, "ny nx"]
            Inverse-variance map for `imap`.
        posmap : Float[ndmap, "2 ny nx"]
            Position map in radians. The first component is `eta` and the
            second is `xi`.
        guess : AxisManager
            Initial parameter values useful for starting the fit. See
            :func:`make_guess` for the standard parameters.
        map_units : str, default: "pW"
            Units of the input map.
        **kwargs : object
            Additional arguments for the specific fitting function.

        Returns
        -------
        fit_params : Optional[AxisManager]
            Fitted model parameters. Returns `None` if the fit fails.
        model : Optional[Float[ndmap, "ny nx"]]
            Model evaluated with the fitted parameters. Returns `None` if
            the fit fails.
        """
        ...


def make_guess(
    amp: float = 1,
    fwhm_xi: float = 2 / 60,
    fwhm_eta: float = 2 / 60,
    xi0: float = 0,
    eta0: float = 0,
    phi: float = 0,
    off: float = 0,
) -> AxisManager:
    """Make an initial beam-fitting parameter guess.

    Parameters
    ----------
    amp : float, default: 1
        Initial beam amplitude.
    fwhm_xi : float, default: 2 / 60
        Initial FWHM in `xi`.
    fwhm_eta : float, default: 2 / 60
        Initial FWHM in `eta`.
    xi0 : float, default: 0
        Initial beam center in `xi`.
    eta0 : float, default: 0
        Initial beam center in `eta`.
    phi : float, default: 0
        Initial beam rotation angle.
    off : float, default: 0
        Initial DC offset.

    Returns
    -------
    guess : AxisManager
        `AxisManager` containing the initial parameters. All values are scalar
        and positional parameters are in radians.
    """
    guess_dict = locals()
    guess = AxisManager()
    for n, v in guess_dict.items():
        guess.wrap(n, v)
    return guess
