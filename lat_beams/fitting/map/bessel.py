r"""
Fit and evaluate a Bessel+multipole beam model with an $r^{-3}$ wing.

The full beam model is:

$$
M(r,\theta)
=
[1-w(r,\theta)]
\left[
b_{\rm core}
+
\sum_{n_0 = 0}{n_{bessel}}
\sum_{n_1 = 0}{n_0}
\frac{J_{n_0}(\ell_{\max}r)}{\ell_{\max}r}
\frac{J_{n_1}(\ell_{\max}r)}{\ell_{\max}r}
\left(
a_{n_0,n_1,0}
+
\sum_{m=1}_{n_{multipole}}
\left[
a_{n_0,n_1,m,c}\cos(m\theta)
+
a_{n_0,n_1,m,s}\sin(m\theta)
\right]
\right)
\right]
+
w(r,\theta)
\frac{\exp[F(\theta)c_a]}{r^3}
+
b,
$$

where

$$
F(\theta)
=
\left[
1,\,
\cos\theta,\,
\sin\theta,\,
\ldots,\,
\cos(N\theta),\,
\sin(N\theta)
\right],
$$

and

$$
r_0(\theta)=\exp[F(\theta)c_{r_0}]
$$

The wing transition is

$$
w(r,\theta)
=
\begin{cases}
0, & r \leq r_0-\delta/2,\\
10u^3-15u^4+6u^5,
    & r_0-\delta/2 < r < r_0+\delta/2,\\
1, & r \geq r_0+\delta/2,
\end{cases}
$$

with

$$
\delta=\frac{0.3r_0}{p},
\qquad
u=\frac{r-(r_0-\delta/2)}{\delta}.
$$

`fit_bessel_map` fits the model using variable projection. For
fixed wing parameters, the Bessel-core coefficients and offsets are
solved with a weighted linear least-squares problem, while the wing
parameters are optimized nonlinearly. The nonlinear fit uses an
analytic Jacobian, Fourier regularization, and a penalty against
negative model values.

If requested, the fit covariance is computed from the joint
Gauss-Newton Hessian, including the linear-nonlinear cross-covariance.
`bessel_profile_covariance` propagates this covariance to radial
profiles and beam window functions.

See the individual function docstrings for details.
"""

from typing import cast

import numba
import numpy as np
import scipy.linalg
import scipy.optimize
from astropy import units as u
from healpy.sphtfunc import beam2bl
from jaxtyping import Float
from numpy.typing import NDArray
from pixell.enmap import ndmap
from sotodlib.core import AxisManager, IndexAxis, LabelAxis

from ...beam_utils import radial_profile_lin
from ..core import bessel_term_cached


@numba.jit(nopython=True, fastmath=True, cache=True)
def fast_wing_transition(
    r_fit: Float[NDArray, "n_pixels"],
    loga_coeff: Float[NDArray, "n_ang"],
    logr_coeff: Float[NDArray, "n_ang"],
    p_val: float,
    F_mat_T: Float[NDArray, "n_pixels n_ang"],
) -> tuple[Float[NDArray, "n_pixels"], Float[NDArray, "n_pixels"]]:
    r"""
    The wing parameterization and transition equations are given in the
    module docstring. This function evaluates them at the supplied radial
    coordinates and Fourier design matrix.

    The transition sharpness is clipped to $0.5 \leq p \leq 20$, and the
    radius and amplitude are floored/capped as needed for numerical stability.

    Parameters
    ----------
    r_fit : Float[NDArray, "n_pixels"]
        Radial coordinates of the fitted pixels, in radians.
    loga_coeff : Float[NDArray, "n_ang"]
        Fourier coefficients of $\log a(\theta)$.
    logr_coeff : Float[NDArray, "n_ang"]
        Fourier coefficients of $\log r_0(\theta)$.
    p_val : float
        Logarithm of the transition sharpness, $\log p$.
    F_mat_T : Float[NDArray, "n_pixels n_ang"]
        Transposed Fourier design matrix with shape
        `(n_pixels, n_fourier)`.

    Returns
    -------
    pure_wing : Float[NDArray, "n_pixels"]
        Unblended wing model $a(\theta)/r^3$ at each fitted pixel.
    wing_weight : Float[NDArray, "n_pixels"]
        Core-to-wing mixing weight, ranging from 0 to 1.
    """
    loga = F_mat_T @ loga_coeff
    logr = F_mat_T @ logr_coeff
    a = np.exp(loga)
    r0 = np.exp(logr)
    p = max(0.5, min(20.0, np.exp(p_val)))

    n_fit = len(r_fit)
    pure_wing = np.empty(n_fit)
    wing_weight = np.empty(n_fit)

    for i in range(n_fit):
        r = max(r_fit[i], 1e-5)
        r0_i = max(r0[i], 1e-4)
        delta = 0.3 * r0_i / p
        r_lo = r0_i - 0.5 * delta
        r_hi = r0_i + 0.5 * delta

        if r <= r_lo:
            weight = 0.0
        elif r >= r_hi:
            weight = 1.0
        else:
            u = (r - r_lo) / delta
            weight = u**3 * (10.0 + u * (-15.0 + 6.0 * u))

        wing_weight[i] = weight
        wing = min(a[i], 1e20) / r**3
        pure_wing[i] = wing if np.isfinite(wing) else 0.0

    return pure_wing, wing_weight


def bessel_beam(
    posmap: Float[ndmap, "2 nx ny"],
    xi0: float,
    eta0: float,
    ell_max: float,
    amps: Float[NDArray, "n_bessel n_bessel n_multipoles 2"],
    bessel_off: float,
    wing_params: Float[NDArray, "n_wing_params"],
    off: float,
) -> Float[ndmap, "nx ny"]:
    r"""
    Evaluate a fitted Bessel-core plus smooth $r^{-3}$-wing beam model.

    The full model and its parameterization are described in the module
    docstring. In particular, the Bessel core is evaluated using the
    normalized terms

    $$
    \frac{J_n(\ell_{\max}r)}{\ell_{\max}r}
    $$

    which are then summed up as the multipole expansion of the pairwise products
    of the Bessel terms.

    The wing and transition is evaluated by `fast_wing_transition`
    using the same implementation as the fitting code.

    Parameters
    ----------
    posmap : Float[ndmap, "2 nx ny"]
        Two-dimensional coordinate maps ``(eta, xi)`` in radians.
    xi0 : float
        Beam center in the ``xi`` coordinate, in radians.
    eta0 : float
        Beam center in the ``eta`` coordinate, in radians.
    ell_max : float
        Maximum multipole used to construct the Bessel basis.
    amps : Float[NDArray, "n_bessel n_bessel n_multipoles 2"]
        Bessel/multipole coefficients. The final axis contains cosine and
        sine coefficients.
    bessel_off : float
        Additive offset applied to the Bessel core.
    wing_params : Float[NDArray, "n_params"]
        Nonlinear wing parameters passed to `fast_wing_transition`.
    off : float
        Global additive model offset.

    Returns
    -------
    ndmap
        Model evaluated on the same two-dimensional grid as ``posmap``.
    """

    n_bessel = amps.shape[0]
    n_multipoles = amps.shape[2]
    n_ang = 1 + 2 * n_multipoles

    eta, xi = posmap
    xi_rel = xi - xi0
    eta_rel = eta - eta0
    theta = np.arctan2(eta_rel, xi_rel)
    r = np.hypot(xi_rel, eta_rel)

    orig_shape = r.shape
    r_flat = r.ravel()
    theta_flat = theta.ravel()
    n_pixels = r_flat.size

    # Fourier basis for the angular dependence.
    F_mat = np.empty((n_ang, n_pixels), dtype=float)
    F_mat[0] = 1.0

    cos_m_theta, sin_m_theta = [], []
    if n_multipoles:
        m = np.arange(1, n_multipoles + 1)[:, None]
        cos_m_theta = np.cos(m * theta_flat)
        sin_m_theta = np.sin(m * theta_flat)

        for i in range(n_multipoles):
            F_mat[1 + 2 * i] = cos_m_theta[i]
            F_mat[2 + 2 * i] = sin_m_theta[i]

    F_mat_T = F_mat.T

    # Construct the Bessel pair basis.
    b_terms = np.column_stack(
        [bessel_term_cached(r_flat, ell_max, n) for n in range(n_bessel)]
    )
    n0, n1 = np.triu_indices(n_bessel)
    base_beams = np.nan_to_num(
        b_terms[:, n0] * b_terms[:, n1],
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Reconstruct the pure Bessel/multipole core.
    core_flat = np.full(n_pixels, bessel_off, dtype=float)

    for pair_idx, (i0, i1) in enumerate(zip(n0, n1)):
        pair_beam = base_beams[:, pair_idx]

        for m_idx in range(n_multipoles):
            c = amps[i0, i1, m_idx, 0]
            if c != 0:
                angular = 1.0 if m_idx == 0 else cos_m_theta[m_idx - 1]
                core_flat += c * pair_beam * angular

            if m_idx > 0:
                s = amps[i0, i1, m_idx, 1]
                if s != 0:
                    core_flat += s * pair_beam * sin_m_theta[m_idx - 1]

    # Evaluate the wing and transition using the same implementation as
    # the fitting code.
    _, wing_weight = fast_wing_transition(
        r_flat,
        wing_params[:n_ang],
        wing_params[n_ang : 2 * n_ang],
        wing_params[-1],
        F_mat_T,
    )

    loga = F_mat_T @ wing_params[:n_ang]
    a = np.exp(loga)

    pure_wing = np.nan_to_num(
        a / np.maximum(r_flat, 1e-5) ** 3,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    model = ((1.0 - wing_weight) * core_flat + wing_weight * pure_wing + off).reshape(
        orig_shape
    )
    return ndmap(model.reshape(orig_shape), posmap.wcs)


def bessel_beam_from_aman(
    posmap: Float[ndmap, "2 nx ny"],
    aman: AxisManager,
) -> Float[ndmap, "nx ny"]:
    """
    Evaluate a fitted Bessel beam from an `AxisManager`.

    Parameters
    ----------
    posmap : Float[ndmap, "2 nx ny"]
        Two-dimensional beam-coordinate maps `(eta, xi)` in radians.
    aman : AxisManager
        Fitted parameter container produced by `fit_bessel_map`.

    Returns
    -------
    beam_model : Float[ndmap, "nx ny"]
        Beam model evaluated on `posmap` using the fitted parameters.
    """
    xi0 = cast(u.Quantity, aman.gauss.xi0)
    eta0 = cast(u.Quantity, aman.gauss.eta0)

    ell_max = cast(float, aman.bessel.ell_max.value)
    amps = cast(NDArray[np.floating], aman.bessel.amps.value)
    bessel_off = cast(float, aman.bessel.bessel_off.value)
    wing_params = cast(NDArray[np.floating], aman.bessel.wing_params.value)
    off = cast(float, aman.bessel.off.value)

    return bessel_beam(
        posmap,
        float(xi0.to(u.radian).value),
        float(eta0.to(u.radian).value),
        ell_max,
        amps,
        bessel_off,
        wing_params,
        off,
    )


def fit_bessel_map(
    imap: Float[ndmap, "nx ny"],
    ivar: Float[ndmap, "nx ny"],
    posmap: Float[ndmap, "2 nx ny"],
    guess: AxisManager,
    map_units: str = "pW",
    n_bessel: int = 10,
    n_multipoles: int = 5,
    d: u.Quantity = 6 * u.m,
    lmd: u.Quantity = 3.3 * u.mm,
    mask_size: float = np.inf,
    n_sigma: float = 5,
    skip_multipoles: list[int] = [],
    calc_cov: bool = False,
    n_opt_pixels: int = 6000,
) -> tuple[AxisManager, Float[ndmap, "nx ny"]]:
    r"""
    Fit a Bessel+multipole core plus a smooth positive $r^{-3}$ wing.

    The detailed model definition is implemented by `bessel_beam`; this
    function is mainly concerned with setting up and solving the fit.

    The fit separates the parameters into nonlinear wing parameters and
    linear Bessel-core coefficients. For fixed nonlinear parameters, the
    model is linear in the Bessel coefficients, so those coefficients are
    solved analytically at every nonlinear iteration. This is a
    variable-projection-style fit: `least_squares` only searches over the
    wing parameters while the linear parameters are eliminated by a small
    weighted least-squares solve.

    The nonlinear parameters are the Fourier coefficients describing the
    wing amplitude and transition radius, plus the logarithm of the
    transition sharpness:

    $$
    \mathbf{p} =
    \left[
        \log a_k,\,
        \log r_{0,k},\,
        \log p
    \right]
    $$

    At each nonlinear iteration the wing parameters are evaluated on the
    optimization pixels. The resulting wing weight is `w_wing`, and the
    weighted design matrix for the linear solve is

    $$
    X =
    \begin{bmatrix}
        (1-w_{\rm wing}) B_1 &
        \cdots &
        (1-w_{\rm wing}) B_{N_{\rm core}} &
        1
    \end{bmatrix}
    $$

    The wing contribution is moved to the right-hand side:

    $$
    \mathbf{y}' = \mathbf{y} - w_{\rm wing}\mathbf{W}
    $$

    The linear coefficients are then found from

    $$
    (X^T X + R)\mathbf{c} = X^T\mathbf{y}'
    $$

    where `R` is a small ridge applied only to the Bessel-core
    coefficients. The solve is performed with a Cholesky factorization.

    The Bessel design matrix is expensive to construct, so it is computed
    once for all valid pixels and then reused throughout the nonlinear
    optimization. In particular, the individual Bessel terms and all
    pairwise products are stored in `base_beams`. The angular Fourier
    factors are also computed once and stored in `cos_m_theta`,
    `sin_m_theta`, and `F_mat_T`.

    The Bessel design matrix is column-scaled before the linear solve.
    If `B` is the unscaled design matrix and `w` contains the pixel
    weights, the stored matrix is

    $$
    B_s =
    \frac{B \odot w}{s},
    \qquad
    s_j =
    \left\|
        (B_{\cdot j} \odot w)
    \right\|_2
    $$

    This keeps the columns of the normal matrix reasonably conditioned.
    The solved coefficients are transformed back to the original
    normalization by dividing by the same scale factors.

    Only `n_opt_pixels` pixels are used during the nonlinear optimization.
    The selection is deterministic and is spread across radial bins and
    approximately uniformly in angle. This is mainly a speed optimization:
    the expensive nonlinear search only needs a representative sample of
    the map. After the nonlinear parameters have converged, the linear
    coefficients are solved one final time using every valid fitting pixel.

    The nonlinear residual contains three pieces: the weighted map
    residual, Fourier regularization, and a penalty against negative
    model values:

    $$
    \mathbf{r} =
    \begin{bmatrix}
        \mathbf{M}_w-\mathbf{y}_w \\
        0.01\,\mathbf{q} \\
        P_{\rm neg}
    \end{bmatrix}
    $$

    The Fourier regularization acts only on the non-constant angular
    coefficients:

    $$
    \mathbf{q} =
    \begin{bmatrix}
        \log a_1,\ldots,\log a_{N-1},
        \log r_{0,1},\ldots,\log r_{0,N-1}
    \end{bmatrix}
    $$

    The negative-model penalty is

    $$
    P_{\rm neg} = 10 \sqrt{ \sum_i \left[ \max(0,-M_i)w_i \right]^2 }
    $$

    An analytic Jacobian is supplied to `scipy.optimize.least_squares`,
    avoiding finite-difference evaluations of the nonlinear residual.

    For the wing amplitude,

    $$
    W = \frac{a}{r^3}
    $$

    and since the amplitude is parameterized as `log(a)`, its derivative
    with respect to a Fourier coefficient is

    $$
    \frac{\partial W_i} {\partial \log a_k} = W_i F_{ik}
    $$

    The transition coordinate can be written as

    $$
    u = \frac{1}{2} + \frac{p}{0.3} \left( \frac{r}{r_0}-1 \right)
    $$

    Inside the transition region the wing weight is the quintic
    smoothstep

    $$
    s(u) = 10u^3 - 15u^4 + 6u^5
    $$

    with derivative

    $$
    \frac{ds}{du} = 30u^2(1-u)^2
    $$

    For the logarithmic transition-radius and sharpness parameters,

    $$
    \frac{\partial u}{\partial\log r_0}
    =
    -\frac{p}{0.3}\frac{r}{r_0}
    $$

    and

    $$
    \frac{\partial u}{\partial\log p}
    =
    \frac{p}{0.3}
    \left(
        \frac{r}{r_0}-1
    \right)
    $$

    Therefore

    $$
    \frac{\partial w_{\rm wing}}
    {\partial\log r_0}
    =
    \frac{ds}{du}
    \frac{\partial u}{\partial\log r_0},
    $$

    and

    $$
    \frac{\partial w_{\rm wing}}
    {\partial\log p}
    =
    \frac{ds}{du}
    \frac{\partial u}{\partial\log p}.
    $$

    For the blended model

    $$
    M =
    (1-w_{\rm wing})C
    +
    w_{\rm wing}W
    +
    {\rm off}
    $$

    the derivative with respect to a transition parameter is

    $$
    \frac{\partial M}{\partial q}
    =
    \frac{\partial w_{\rm wing}}{\partial q}
    (W-C)
    +
    w_{\rm wing}
    \frac{\partial W}{\partial q}
    $$

    The Jacobian returned to `least_squares` is the Jacobian of the
    residual, so the map-model derivatives are additionally multiplied
    by the pixel weights. The regularization rows have derivatives
    `0.01` with respect to their corresponding Fourier coefficients.

    The linear coefficients are held fixed when constructing this
    Jacobian. In other words, the Jacobian differentiates the current
    profiled model while not differentiating through the Cholesky solve.
    This keeps the Jacobian cheap and avoids having to differentiate the
    variable-projection solve itself.

    If `calc_cov=True`, the nonlinear covariance is estimated from the
    Jacobian at the final solution:

    $$
    C_{\rm wing}
    =
    \sigma^2
    (J^T J)^+
    $$

    where `+` denotes the pseudoinverse and

    $$
    \sigma^2 =
    \frac{\chi^2}{N_{\rm dof}}
    $$

    The linear covariance is computed from the final full-pixel Cholesky
    factorization:

    $$
    C_{\rm linear,scaled}
    =
    \sigma^2
    (X^T X + R)^{-1}
    $$

    and transformed back from the column-scaled basis using

    $$
    C_{\rm linear}
    =
    D^{-1}
    C_{\rm linear,scaled}
    D^{-1}
    $$

    where `D` is the diagonal matrix of column scales.

    Parameters
    ----------
    imap : Float[ndmap, "nx ny"]
        Input map to fit.
    ivar : Float[ndmap, "nx ny"]
        Inverse-variance map corresponding to `imap`.
    posmap : Float[ndmap, "2 nx ny"]
        Position maps `(eta, xi)` in radians.
    guess : AxisManager
        Initial source position, FWHM, and offset estimates.
    map_units : str, optional
        Units of the input map and fitted linear coefficients.
    n_bessel : int, optional
        Number of Bessel basis functions.
    n_multipoles : int, optional
        Number of angular multipoles.
    d : u.Quantity, optional
        Telescope aperture diameter.
    lmd : u.Quantity, optional
        Wavelength corresponding to the observation frequency.
    mask_size : float, optional
        Radius of the fitting region in radians.
    n_sigma : float, optional
        Threshold used when estimating the initial transition radius.
    skip_multipoles : list[int], optional
        Multipoles omitted from the linear Bessel model.
    calc_cov : bool, optional
        Calculate nonlinear and linear covariance estimates.
    n_opt_pixels : int, optional
        Number of representative pixels used during nonlinear
        optimization. Set to `None` to use all valid pixels.

    Returns
    -------
    aman : AxisManager
        Fitted parameters and diagnostics. The wrapped fields are:

        - `ell_max`: scalar dimensionless value.
        - `mask_size`: scalar quantity with units of radians.
        - `n_sigma`: scalar dimensionless value.
        - `bessel_idx`: `(n_linear_core, 4)` integer array containing
          `(n0, n1, m, sin/cos)` indices.
        - `amps`: `(n_bessel, n_bessel, n_multipoles, 2)` array with map
          units.
        - `bessel_off`: scalar map-unit offset.
        - `off`: scalar map-unit global offset.
        - `wing_params`: `(2 * n_ang + 1,)` dimensionless nonlinear
          parameter vector.
        - `wing_p`: scalar dimensionless transition sharpness.
        - `chi2`: scalar fit statistic.
        - `dof`: scalar number of degrees of freedom.
        - `wing_success`: scalar boolean optimization status.
        - `n_opt_pixels`: scalar number of optimization pixels.
        - `n_fit_pixels`: scalar number of valid fitting pixels.

        If `calc_cov=True`, the following are also wrapped:

        - `wing_cov`: `(2 * n_ang + 1, 2 * n_ang + 1)` nonlinear
          covariance matrix.
        - `wing_errors`: `(2 * n_ang + 1,)` nonlinear parameter errors.
        - `core_cov`: `(n_linear, n_linear)` linear covariance matrix.

    beam_model : Float[ndmap, "nx ny"]
        Final fitted model evaluated on the full input position map.
    """
    ell_max = np.pi * (d / lmd).decompose().value

    eta, xi = posmap
    eta0 = float(cast(u.Quantity, guess.eta0).to(u.radian).value)
    xi0 = float(cast(u.Quantity, guess.xi0).to(u.radian).value)

    xi = xi - xi0
    eta = eta - eta0

    theta = np.arctan2(eta, xi)
    r = np.hypot(xi, eta)

    fwhm = 0.5 * (
        float(cast(u.Quantity, guess.fwhm_xi).to(u.radian).value)
        + float(cast(u.Quantity, guess.fwhm_eta).to(u.radian).value)
    )

    ivar = ivar.copy()
    ivar[r == 0] = 0
    map_unit = u.Unit(map_units)

    aman = AxisManager()
    bessel_ax = IndexAxis("bessel", n_bessel)
    multipole_ax = IndexAxis("multipole", n_multipoles)
    term_ax = LabelAxis("term", ["cos", "sin"])

    aman.wrap("ell_max", ell_max * u.dimensionless_unscaled)
    aman.wrap("mask_size", mask_size * u.radian)
    aman.wrap("n_sigma", n_sigma * u.dimensionless_unscaled)

    # Full fitting mask.
    fit_msk = (
        np.isfinite(imap)
        * np.isfinite(ivar)
        * (ivar > 0)
        * (r < 1.5 * mask_size)
        * (r > 1e-12)
    )

    r_fit = np.asarray(r[fit_msk], dtype=float)
    theta_fit = np.asarray(theta[fit_msk], dtype=float)
    map_fit = np.asarray(imap[fit_msk], dtype=float)
    w = np.sqrt(np.asarray(ivar[fit_msk], dtype=float))
    y = map_fit * w

    n_pixels = len(r_fit)

    if n_pixels == 0:
        raise ValueError("No valid pixels found for Bessel fit.")

    # Background.
    outer_mask = (r > mask_size) * np.isfinite(imap) * np.isfinite(ivar) * (ivar > 0)

    if np.any(outer_mask):
        background = np.median(np.asarray(imap[outer_mask], dtype=float))
    else:
        background = guess.off.value

    # Multipole bookkeeping.
    skip_set = set(skip_multipoles)
    n_fourier = n_multipoles
    n_ang = 1 + 2 * n_fourier
    m_arr = np.arange(1, n_fourier + 1)

    cos_m_theta = np.cos(m_arr[:, None] * theta_fit)
    sin_m_theta = np.sin(m_arr[:, None] * theta_fit)

    triu_indices = np.triu_indices(n_bessel)

    # Construct the Bessel basis once for all fitting pixels.
    b_terms = np.column_stack(
        [
            cast(NDArray[np.floating], bessel_term_cached(r_fit, ell_max, n))
            for n in range(n_bessel)
        ]
    )
    base_beams = b_terms[:, triu_indices[0]] * b_terms[:, triu_indices[1]]

    base_beams = np.nan_to_num(
        base_beams,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Construct the unweighted core design matrix.
    idx_list = []
    X_cols = []

    for pair_idx, (n0, n1) in enumerate(zip(triu_indices[0], triu_indices[1])):
        base_beam = base_beams[:, pair_idx]

        for m in range(n_multipoles):
            if m in skip_set:
                continue

            if m == 0:
                idx_list.append((n0, n1, 0, 0))
                X_cols.append(base_beam)
            else:
                idx_list.append((n0, n1, m, 0))
                X_cols.append(base_beam * cos_m_theta[m - 1])
                idx_list.append((n0, n1, m, 1))
                X_cols.append(base_beam * sin_m_theta[m - 1])

    B_core = np.column_stack(X_cols)
    n_core = B_core.shape[1]
    B = np.column_stack([B_core, np.ones_like(r_fit)])
    n_linear = B.shape[1]

    core_names = [
        f"n{n0}_n{n1}_m{m}_{'sin' if term else 'cos'}" for n0, n1, m, term in idx_list
    ]
    linear_names = core_names + ["offset"]
    core_ax = LabelAxis("core", core_names)
    linear_ax = LabelAxis("linear", linear_names)
    bessel_index_ax = LabelAxis("bessel_index", ["n0", "n1", "m", "term"])
    wing_names = ["loga_m0"] + [
        f"loga_m{m}_{t}" for m in range(1, n_fourier + 1) for t in ("cos", "sin")
    ]
    wing_names += ["logr_m0"] + [
        f"logr_m{m}_{t}" for m in range(1, n_fourier + 1) for t in ("cos", "sin")
    ]
    wing_names += ["log_p"]
    wing_ax = LabelAxis("wing_param", wing_names)
    fit_ax = LabelAxis("fit_param", linear_names + wing_names)

    # Column scaling.
    B_weighted = B * w[:, None]
    scale = np.linalg.norm(B_weighted, axis=0)
    scale[scale == 0] = 1.0
    Bs = B_weighted / scale

    # Fourier matrix for the wing parameters.
    F_mat = np.zeros((n_ang, n_pixels))
    F_mat[0] = 1.0

    for idx_f, m in enumerate(range(1, n_fourier + 1)):
        F_mat[1 + 2 * idx_f] = cos_m_theta[idx_f]
        F_mat[2 + 2 * idx_f] = sin_m_theta[idx_f]

    F_mat_T = F_mat.T.copy()
    aman.wrap(
        "bessel_idx",
        np.asarray(idx_list, dtype=int),
        [(0, core_ax), (1, bessel_index_ax)],
    )

    # Choose optimization pixels.
    if n_opt_pixels is None or n_opt_pixels >= n_pixels:
        opt_idx = np.arange(n_pixels)
    else:
        n_rbins = 60
        n_per_bin = max(1, int(np.ceil(n_opt_pixels / n_rbins)))
        r_edges = np.linspace(
            np.min(r_fit),
            np.max(r_fit),
            n_rbins + 1,
        )
        opt_chunks = []
        for ibin in range(n_rbins):
            sel = np.flatnonzero(
                (r_fit >= r_edges[ibin])
                * (
                    r_fit < r_edges[ibin + 1]
                    if ibin < n_rbins - 1
                    else r_fit <= r_edges[ibin + 1]
                )
            )
            if len(sel) == 0:
                continue
            if len(sel) <= n_per_bin:
                opt_chunks.append(sel)
                continue
            order = sel[np.argsort(theta_fit[sel])]
            take = np.linspace(
                0,
                len(order) - 1,
                n_per_bin,
            ).astype(int)
            opt_chunks.append(order[take])
        opt_idx = np.concatenate(opt_chunks)
        if len(opt_idx) > n_opt_pixels:
            take = np.linspace(
                0,
                len(opt_idx) - 1,
                n_opt_pixels,
            ).astype(int)
            opt_idx = opt_idx[take]
    opt_idx = np.unique(opt_idx)

    # Optimization subset.
    r_opt = r_fit[opt_idx]
    theta_opt = theta_fit[opt_idx]
    w_opt = w[opt_idx]
    y_opt = y[opt_idx]
    B_core_opt = B_core[opt_idx]
    Bs_opt = Bs[opt_idx]
    F_mat_T_opt = F_mat_T[opt_idx]

    # Initial transition radius.
    r0_init = min(0.9 * mask_size, 3.0 * fwhm)

    valid_profile = (
        np.isfinite(map_fit)
        * np.isfinite(w)
        * (map_fit != 0)
        * (r_fit < mask_size)
        * (r_fit > fwhm)
    )

    if np.any(valid_profile):
        r_prof = r_fit[valid_profile]
        y_prof = map_fit[valid_profile]

        sort_r = np.sort(np.unique(r_fit))
        pixel_size = np.median(np.diff(sort_r)) if len(sort_r) > 1 else 0.05 * fwhm

        bin_size = 4.0 * pixel_size
        r_min = np.min(r_prof)
        r_max = np.max(r_prof)

        bins = np.arange(r_min, r_max + bin_size, bin_size)
        if len(bins) > 2:
            digitized = np.digitize(r_prof, bins)
            r_binned = []
            y_binned = []
            for i in range(1, len(bins)):
                bin_mask = digitized == i
                if np.any(bin_mask):
                    r_binned.append(np.median(r_prof[bin_mask]))
                    y_binned.append(np.median(y_prof[bin_mask]))
            r_prof_clean = np.asarray(r_binned)
            y_prof_clean = np.asarray(y_binned)
        else:
            r_prof_clean = r_prof
            y_prof_clean = y_prof
        central = np.asarray(
            imap[(r < fwhm) * np.isfinite(imap)],
            dtype=float,
        )
        if central.size:
            peak_val = np.max(central)
        else:
            peak_val = np.nanmax(np.asarray(imap))
        threshold = 10.0 ** (-n_sigma) * peak_val
        below_threshold = r_prof_clean[y_prof_clean <= threshold]
        if len(below_threshold):
            r0_init = np.min(below_threshold)
        r0_init = np.clip(
            r0_init,
            1.2 * fwhm,
            0.9 * mask_size,
        )

    # Initial wing amplitude.
    annulus_mask = (r_fit >= 0.95 * r0_init) * (r_fit <= 1.05 * r0_init)
    if np.any(annulus_mask):
        y_at_r0 = np.median(map_fit[annulus_mask]) - background
        y_at_r0 = max(y_at_r0, 0.0)
    else:
        y_at_r0 = 0.0

    if y_at_r0 <= 0:
        central = np.asarray(
            imap[(r < fwhm) * np.isfinite(imap)],
            dtype=float,
        )
        if central.size:
            y_at_r0 = max(np.max(central) - background, 0.0) * 1e-3
    a_init = max(y_at_r0 * r0_init**3, 1e-15)

    params0 = np.zeros(2 * n_ang + 1)
    params0[0] = np.log(a_init)
    params0[n_ang] = np.log(r0_init)
    params0[-1] = np.log(3.0)

    # Bounds.
    lower = np.full_like(params0, -np.inf)
    upper = np.full_like(params0, np.inf)
    lower[n_ang] = np.log(max(1.2 * fwhm, 1e-8))
    if np.isfinite(mask_size):
        upper[n_ang] = np.log(max(0.9 * mask_size, 1.2 * fwhm))

    lower[-1] = np.log(0.5)
    upper[-1] = np.log(20.0)

    def solve_linear_opt(params):
        """
        Solve the profiled linear problem on the optimization pixels.
        """
        loga_coeff, logr_coeff, p_val = (
            params[:n_ang],
            params[n_ang : 2 * n_ang],
            params[-1],
        )

        pure_wing, wing_weight = fast_wing_transition(
            r_opt, loga_coeff, logr_coeff, p_val, F_mat_T_opt
        )
        q = 1.0 - wing_weight

        X = np.column_stack([Bs_opt[:, :-1] * q[:, None], Bs_opt[:, -1]])
        wing_weighted = pure_wing * wing_weight * w_opt
        rhs = y_opt - wing_weighted

        A = X.T @ X
        A[:n_core, :n_core] += 1e-6 * np.eye(n_core)
        b = X.T @ rhs

        cho = scipy.linalg.cho_factor(A, lower=True, check_finite=False)
        linear_scaled = scipy.linalg.cho_solve(cho, b, check_finite=False)

        linear = linear_scaled / scale
        pure_core = B_core_opt @ linear[:-1]
        global_offset = linear[-1]
        model = q * pure_core + wing_weight * pure_wing + global_offset

        return linear, model * w_opt, pure_core, pure_wing, wing_weight

    def jacobian(params):
        """
        Calculate the analytic Jacobian.
        """
        _, _, pure_core, pure_wing, wing_weight = solve_linear_opt(params)

        loga_coeff = params[:n_ang]
        logr_coeff = params[n_ang : 2 * n_ang]
        p_val = params[-1]

        n_data = len(r_opt)
        loga = F_mat_T_opt @ loga_coeff
        logr = F_mat_T_opt @ logr_coeff
        a = np.exp(loga)
        r0 = np.exp(logr)

        p = np.clip(np.exp(p_val), 0.5, 20.0)
        rr = np.maximum(r_opt, 1e-5)
        r0_safe = np.maximum(r0, 1e-4)

        ratio = rr / r0_safe
        u = 0.5 + (p / 0.3) * (ratio - 1.0)
        inside = (u > 0.0) * (u < 1.0)

        dsdu = np.zeros_like(u)
        ui = u[inside]
        dsdu[inside] = 30.0 * ui**2 * (1.0 - ui) ** 2

        du_dlogr0 = -(p / 0.3) * ratio
        du_dlogp = (p / 0.3) * (ratio - 1.0)
        dw_dlogr0 = dsdu * du_dlogr0
        dw_dlogp = dsdu * du_dlogp
        dw_dlogr0[~inside] = 0.0
        dw_dlogp[~inside] = 0.0

        dwing_dloga = pure_wing[:, None] * F_mat_T_opt
        wing_minus_core = pure_wing - pure_core

        n_params = len(params)
        n_reg = 2 * (n_ang - 1)
        J = np.zeros((n_data + n_reg + 1, n_params), dtype=float)

        dmodel_a = wing_weight[:, None] * dwing_dloga
        J[:n_data, :n_ang] = dmodel_a * w_opt[:, None]

        dmodel_r0 = wing_minus_core[:, None] * dw_dlogr0[:, None] * F_mat_T_opt
        J[:n_data, n_ang : 2 * n_ang] = dmodel_r0 * w_opt[:, None]

        dmodel_p = wing_minus_core * dw_dlogp
        J[:n_data, -1] = dmodel_p * w_opt

        reg_start = n_data
        J[reg_start : reg_start + n_ang - 1, 1:n_ang] = 0.01 * np.eye(n_ang - 1)

        reg_start += n_ang - 1
        J[
            reg_start : reg_start + n_ang - 1,
            n_ang + 1 : 2 * n_ang,
        ] = 0.01 * np.eye(n_ang - 1)

        model_no_offset = (1.0 - wing_weight) * pure_core + wing_weight * pure_wing
        negative = np.maximum(0.0, -model_no_offset)
        neg_norm = np.sqrt(np.sum((negative * w_opt) ** 2))

        if neg_norm > 0.0:
            dmodel_no_offset = np.zeros((n_data, n_params), dtype=float)
            dmodel_no_offset[:, :n_ang] = dmodel_a
            dmodel_no_offset[:, n_ang : 2 * n_ang] = dmodel_r0
            dmodel_no_offset[:, -1] = dmodel_p

            active = model_no_offset < 0.0
            weights = np.zeros_like(model_no_offset)
            weights[active] = negative[active] * w_opt[active] ** 2

            J[-1, :] = (
                -10.0 * (weights[:, None] * dmodel_no_offset).sum(axis=0) / neg_norm
            )

        return J

    def residual(params):
        """
        Evaluate the nonlinear least-squares residual.
        """
        _, model_weighted, pure_core, pure_wing, wing_weight = solve_linear_opt(params)

        res = model_weighted - y_opt
        loga_coeff = params[:n_ang]
        logr_coeff = params[n_ang : 2 * n_ang]

        reg_array = 0.01 * np.concatenate([loga_coeff[1:], logr_coeff[1:]])

        model_no_offset = (1.0 - wing_weight) * pure_core + wing_weight * pure_wing
        negative = np.maximum(0.0, -model_no_offset)

        neg_penalty = np.array([10.0 * np.sqrt(np.sum((negative * w_opt) ** 2))])

        return np.nan_to_num(
            np.concatenate([res, reg_array, neg_penalty]),
            nan=1e5,
            posinf=1e5,
            neginf=-1e5,
        )

    # Nonlinear optimization.
    result = scipy.optimize.least_squares(
        residual,
        params0,
        method="trf",
        bounds=(lower, upper),
        max_nfev=60,
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
        jac=jacobian,  # type: ignore
    )

    def solve_linear_full(params):
        """
        Solve the profiled linear problem using all fitting pixels.
        """
        loga_coeff = params[:n_ang]
        logr_coeff = params[n_ang : 2 * n_ang]
        p_val = params[-1]

        pure_wing, wing_weight = fast_wing_transition(
            r_fit, loga_coeff, logr_coeff, p_val, F_mat_T
        )
        q = 1.0 - wing_weight

        X = np.column_stack([Bs[:, :-1] * q[:, None], Bs[:, -1]])
        wing_weighted = pure_wing * wing_weight * w
        rhs = y - wing_weighted

        A = X.T @ X
        A[:n_core, :n_core] += 1e-6 * np.eye(n_core)
        b = X.T @ rhs

        cho = scipy.linalg.cho_factor(A, lower=True, check_finite=False)
        linear_scaled = scipy.linalg.cho_solve(cho, b, check_finite=False)

        linear = linear_scaled / scale
        pure_core = B_core @ linear[:-1]
        global_offset = linear[-1]
        model = q * pure_core + wing_weight * pure_wing + global_offset

        return linear, model * w, pure_core, pure_wing, wing_weight, cho

    linear, *_ = solve_linear_full(result.x)
    coeff, off = linear, linear[-1]

    amps = np.zeros((n_bessel, n_bessel, n_multipoles, 2))
    for idx, value in zip(idx_list, coeff[:-1]):
        amps[idx] = value

    aman.wrap(
        "amps",
        amps * map_unit,
        [(0, bessel_ax), (1, bessel_ax), (2, multipole_ax), (3, term_ax)],
    )
    aman.wrap("linear_coeffs", coeff * map_unit, [(0, linear_ax)])
    aman.wrap("off", off * map_unit)
    aman.wrap("bessel_off", 0 * map_unit)  # Keep for compatibility.

    aman.wrap("wing_params", result.x * u.dimensionless_unscaled, [(0, wing_ax)])
    aman.wrap("wing_p", np.exp(result.x[-1]) * u.dimensionless_unscaled)

    aman.wrap("chi2", np.sum(result.fun**2))
    aman.wrap("dof", max(1, len(result.fun) - len(result.x)))
    aman.wrap("wing_success", result.success)
    aman.wrap("n_opt_pixels", len(opt_idx))
    aman.wrap("n_fit_pixels", n_pixels)
    aman.wrap("n_core", n_core)
    aman.wrap("xi0", xi0 * u.radian)
    aman.wrap("eta0", eta0 * u.radian)

    if calc_cov:
        dof = max(1, len(result.fun) - len(result.x))
        sigma2 = np.sum(result.fun**2) / dof

        full_cov = compute_full_covariance(
            params=result.x,
            linear=linear,
            B_core=B_core,
            w=w,
            F_mat_T=F_mat_T,
            r_fit=r_fit,
            scale=scale,
            n_core=n_core,
            n_ang=n_ang,
            n_pixels=n_pixels,
            sigma2=sigma2,
        )

        n_linear = n_core + 1
        core_cov = full_cov[:n_linear, :n_linear]
        wing_cov = full_cov[n_linear:, n_linear:]
        core_wing_cov = full_cov[:n_linear, n_linear:]
        wing_errors = np.sqrt(np.maximum(0.0, np.diag(wing_cov)))

        aman.wrap("full_cov", full_cov, [(0, fit_ax), (1, fit_ax)])
        aman.wrap("core_cov", core_cov, [(0, linear_ax), (1, linear_ax)])
        aman.wrap("wing_cov", wing_cov, [(0, wing_ax), (1, wing_ax)])
        aman.wrap("core_wing_cov", core_wing_cov, [(0, linear_ax), (1, wing_ax)])
        aman.wrap("wing_errors", wing_errors, [(0, wing_ax)])
        aman.wrap("sigma2", sigma2)

    beam_model = bessel_beam(
        posmap,
        xi0,
        eta0,
        ell_max,
        amps,
        0,
        result.x,
        off,
    )

    return aman, beam_model


def compute_full_covariance(
    params: Float[NDArray, "n_nonlinear"],
    linear: Float[NDArray, "n_linear"],
    B_core: Float[NDArray, "n_pixels n_core"],
    w: Float[NDArray, "n_pixels"],
    F_mat_T: Float[NDArray, "n_pixels n_ang"],
    r_fit: Float[NDArray, "n_pixels"],
    scale: Float[NDArray, "n_linear"],
    n_core: int,
    n_ang: int,
    n_pixels: int,
    sigma2: float,
) -> Float[NDArray, "n_params n_params"]:
    r"""
    Calculate the joint covariance of the linear and nonlinear fit
    parameters, including their cross-covariance.

    The fitted model can be written as

    $$
    M = (1 - w_{\rm wing}) C + w_{\rm wing} W + {\rm off}
    $$

    where `C` is the Bessel core, `W` is the pure $r^{-3}$ wing, and
    `w_wing` is the smooth transition supplied by
    `fast_wing_transition`.

    The linear parameters are the Bessel coefficients and global offset,
    while the nonlinear parameters are the Fourier coefficients describing
    the wing amplitude and transition radius, together with `log(p)`.

    At the final solution the linear parameters are conditionally
    optimized for the nonlinear parameters. For covariance estimation we
    construct the joint Gauss--Newton Hessian using the partial derivatives
    of the model with respect to all parameters,

    $$
    H = J^T J
    $$

    together with the same Fourier regularization and linear ridge used
    during the fit.

    The covariance is then

    $$
    C_{\rm full} = \sigma^2 H^{-1}
    $$

    In block form,

    $$
    H =
    \begin{pmatrix}
        H_{cc} & H_{cq} \\
        H_{cq}^T & H_{qq}
    \end{pmatrix}
    $$

    so the resulting covariance contains the linear/nonlinear
    cross-covariance

    $$
    C_{cq} =
    \operatorname{Cov}(c, q)
    $$

    This is different from simply combining `core_cov` and `wing_cov`,
    which implicitly assumes `C_cq = 0`.

    Parameters
    ----------
    params : Float[NDArray, "n_nonlinear"]
        Final nonlinear parameters. Shape `(2 * n_ang + 1,)`.
    linear : Float[NDArray, "n_linear"]
        Final full-pixel linear solution containing the Bessel coefficients
        followed by the global offset. Shape `(n_linear,)`.
    B_core : Float[NDArray, "n_pixels n_core"]
        Unweighted Bessel design matrix. Shape `(n_pixels, n_core)`.
    w : Float[NDArray, "n_pixels"]
        Pixel weights, equal to the square root of the inverse variance.
        Shape `(n_pixels,)`.
    F_mat_T : Float[NDArray, "n_pixels n_ang"]
        Fourier design matrix transposed. Shape `(n_pixels, n_ang)`.
    r_fit : Float[NDArray, "n_pixels"]
        Radius of each fitted pixel. Shape `(n_pixels,)`.
    scale : Float[NDArray, "n_linear"]
        Column scaling used for the linear solve. Shape `(n_linear,)`.
    n_core : int
        Number of Bessel-core linear coefficients.
    n_ang : int
        Number of Fourier angular coefficients.
    n_pixels : int
        Number of fitted pixels.
    sigma2 : float
        Residual variance used to scale the parameter covariance.

    Returns
    -------
    full_cov : Float[NDArray, "n_params n_params"]
        Joint covariance of the linear and nonlinear parameters. Shape
        `(n_linear + 2 * n_ang + 1, n_linear + 2 * n_ang + 1)`.
    """
    # TODO: The regularization and penalization terms have constants that are hardcoded in a few spots, need to fix that
    loga_coeff = params[:n_ang]
    logr_coeff = params[n_ang : 2 * n_ang]
    p_val = params[-1]

    pure_wing, wing_weight = fast_wing_transition(
        r_fit,
        loga_coeff,
        logr_coeff,
        p_val,
        F_mat_T,
    )

    q = 1.0 - wing_weight
    core = B_core @ linear[:-1]

    n_linear = n_core + 1
    n_nonlinear = 2 * n_ang + 1
    n_params = n_linear + n_nonlinear

    X = np.column_stack(
        [
            B_core * q[:, None],
            np.ones(n_pixels),
        ]
    )
    X_weighted = X * w[:, None]

    logr = F_mat_T @ logr_coeff
    r0 = np.exp(logr)

    p = np.clip(
        np.exp(p_val),
        0.5,
        20.0,
    )
    rr = np.maximum(
        r_fit,
        1e-5,
    )
    r0_safe = np.maximum(
        r0,
        1e-4,
    )
    ratio = rr / r0_safe
    u = 0.5 + (p / 0.3) * (ratio - 1.0)

    inside = (u > 0.0) * (u < 1.0)
    dsdu = np.zeros_like(u)
    ui = u[inside]
    dsdu[inside] = 30.0 * ui**2 * (1.0 - ui) ** 2
    du_dlogr0 = -(p / 0.3) * ratio
    du_dlogp = (p / 0.3) * (ratio - 1.0)
    dw_dlogr0 = dsdu * du_dlogr0
    dw_dlogp = dsdu * du_dlogp
    dw_dlogr0[~inside] = 0.0
    dw_dlogp[~inside] = 0.0
    dwing_dloga = pure_wing[:, None] * F_mat_T
    wing_minus_core = pure_wing - core
    dmodel_a = wing_weight[:, None] * dwing_dloga
    dmodel_r0 = wing_minus_core[:, None] * dw_dlogr0[:, None] * F_mat_T
    dmodel_p = wing_minus_core * dw_dlogp
    J_nonlinear = np.column_stack(
        [
            dmodel_a,
            dmodel_r0,
            dmodel_p,
        ]
    )
    J_nonlinear *= w[:, None]

    J = np.zeros(
        (
            n_pixels,
            n_params,
        ),
        dtype=float,
    )
    J[:, :n_linear] = X_weighted
    J[:, n_linear:] = J_nonlinear
    H = J.T @ J

    ridge = 1e-6 / scale[:n_core] ** 2
    H[
        :n_core,
        :n_core,
    ] += np.diag(ridge)
    reg = 0.01**2

    H[
        n_linear + 1 : n_linear + n_ang,
        n_linear + 1 : n_linear + n_ang,
    ] += reg * np.eye(n_ang - 1)
    H[
        n_linear + n_ang + 1 : n_linear + 2 * n_ang,
        n_linear + n_ang + 1 : n_linear + 2 * n_ang,
    ] += reg * np.eye(n_ang - 1)

    model_no_offset = (1.0 - wing_weight) * core + wing_weight * pure_wing
    negative = np.maximum(
        0.0,
        -model_no_offset,
    )
    neg_norm = np.sqrt(np.sum((negative * w) ** 2))
    if neg_norm > 0.0:
        dmodel = np.zeros(
            (
                n_pixels,
                n_params,
            ),
            dtype=float,
        )
        dmodel[:, :n_linear] = X_weighted
        dmodel[
            :,
            n_linear : n_linear + n_nonlinear,
        ] = J_nonlinear
        active = model_no_offset < 0.0
        weights = np.zeros_like(model_no_offset)
        weights[active] = negative[active] * w[active] ** 2
        dpenalty = -10.0 / neg_norm * (weights[:, None] * dmodel).sum(axis=0)
        H += np.outer(
            dpenalty,
            dpenalty,
        )

    covariance = scipy.linalg.pinvh(
        H,
        check_finite=False,
    )

    return sigma2 * cast(NDArray[np.floating], covariance)


def bessel_profile_covariance(
    fit: AxisManager,
    posmap: Float[ndmap, "2 nx ny"],
    lmax: int,
    n_modes: int,
    n_radial: int = 200,
) -> tuple[AxisManager, Float[ndmap, "nx ny"]]:
    r"""
    Propagate the full fit and covariance into a radial beam profile and b_l and their covariances.
    The covariance is propagated using a linearized model,

    $$
    C_y = J_y C_p J_y^T
    $$

    where `p` is the vector of fitted parameters and `y` is either the
    radial profile or the window function.

    If `n_modes` is specified, the fit covariance is approximated as

    $$
    C_p \simeq V_N \Lambda_N V_N^T
    $$

    where the columns of `V_N` are the `n_modes` eigenvectors with the
    largest eigenvalues. The propagated covariance is then

    $$
    C_y \simeq (J_y V_N)\Lambda_N(J_y V_N)^T
    $$

    Parameters
    ----------
    fit : AxisManager
        Output from `fit_bessel_map`. Must contain `full_cov`, `amps`,
        `wing_params`, and `off`.
    posmap : Float[ndmap, "2 nx ny"]
        Beam-coordinate maps `(eta, xi)` used to evaluate the model.
    lmax : int
        Maximum multipole for the window function.
    n_modes : int
        Number of largest covariance modes to retain. If <= 0, all modes
        are retained. The nominal profile and window function are always
        computed from the full best-fit model.
    n_radial : int
        Number of radial bins used for the profile.

    Returns
    -------
    prof_cov : AxisManager
        AxisManager containing the normalized radial profile, beam window
        function, covariance eigenmodes, and normalization information.
    model_no_off : Float[ndmap, "nx ny"]
        The model evaluated at posmap without the offset.
    """
    full_cov = np.asarray(fit.full_cov)
    wing_params = np.asarray(fit.wing_params.value)
    linear = np.asarray(fit.linear_coeffs.value)
    amps = np.asarray(fit.amps.value)
    off = float(np.asarray(fit.off.value))
    p = np.concatenate([linear, wing_params])

    if full_cov.shape != (len(p), len(p)):
        raise ValueError(
            f"full_cov has shape {full_cov.shape}, but the reconstructed parameter vector has length {len(p)}."
        )

    evals, evecs = np.linalg.eigh(full_cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    evals = np.maximum(evals, 0.0)
    if n_modes > 0:
        n_modes = min(n_modes, len(evals))
        evals, evecs = evals[:n_modes], evecs[:, :n_modes]

    xi0 = float(cast(u.Quantity, fit.xi0).to(u.rad).value)
    eta0 = float(cast(u.Quantity, fit.eta0).to(u.rad).value)
    ell_max = float(cast(u.Quantity, fit.ell_max).value)

    model = bessel_beam(posmap, xi0, eta0, ell_max, amps, 0.0, wing_params, off)
    model_flat = np.asarray(model).ravel()
    radial_centers, profile, R = radial_profile_lin(
        model - off,
        posmap,
        xi0=xi0,
        eta0=eta0,
        n_bins=n_radial,
    )

    n_pix = len(model_flat)
    n_modes_actual = len(evals)
    J_mode = np.empty((n_pix, n_modes_actual), dtype=float)
    idx = np.asarray(fit.bessel_idx)
    n_core = int(cast(int, fit.n_core))

    for i, (direction, eval_i) in enumerate(zip(evecs.T, evals)):
        sigma = np.sqrt(eval_i)
        if sigma == 0:
            J_mode[:, i] = 0.0
            continue
        p_plus, p_minus = p + sigma * direction, p - sigma * direction
        amps_plus, amps_minus = np.zeros_like(amps), np.zeros_like(amps)
        for a, ind in zip(p_plus[:n_core], idx):
            amps_plus[tuple(ind)] = a
        for a, ind in zip(p_minus[:n_core], idx):
            amps_minus[tuple(ind)] = a
        model_plus = bessel_beam(
            posmap,
            xi0,
            eta0,
            ell_max,
            amps_plus,
            0.0,
            p_plus[n_core + 1 :],
            p_plus[n_core],
        )
        model_minus = bessel_beam(
            posmap,
            xi0,
            eta0,
            ell_max,
            amps_minus,
            0.0,
            p_minus[n_core + 1 :],
            p_minus[n_core],
        )
        J_mode[:, i] = (
            np.asarray(model_plus).ravel() - np.asarray(model_minus).ravel()
        ) / (2.0 * sigma)

    J_profile_mode = R @ J_mode

    profile_norm = np.max(model) - off
    if not np.isfinite(profile_norm) or profile_norm == 0:
        raise ValueError(f"Invalid profile normalization: {profile_norm}")

    profile /= profile_norm
    J_profile_mode /= profile_norm
    profile_cov = (J_profile_mode * evals[None, :]) @ J_profile_mode.T
    profile_sigma = np.sqrt(np.maximum(np.diag(profile_cov), 0.0))

    ell = np.arange(lmax + 1)
    bl = np.asarray(beam2bl(profile, radial_centers, lmax=lmax))
    bl_mode = np.empty((len(ell), n_modes_actual), dtype=float)

    for i, (mode, eval_i) in enumerate(zip(J_profile_mode.T, evals)):
        bl_mode[:, i] = (
            0.0 if eval_i <= 0 else np.asarray(beam2bl(mode, radial_centers, lmax=lmax))
        )

    bl_cov = (bl_mode * evals[None, :]) @ bl_mode.T
    bl_sigma = np.sqrt(np.maximum(np.diag(bl_cov), 0.0))

    axes = [
        IndexAxis("r", n_radial),
        IndexAxis("ell", len(ell)),
        IndexAxis("mode", n_modes_actual),
        IndexAxis("param", len(p)),
    ]
    out = AxisManager(*axes)

    out.wrap("r", 3600 * np.rad2deg(radial_centers), [(0, "r")])
    out.wrap("profile", profile, [(0, "r")])
    out.wrap("profile_sigma", profile_sigma, [(0, "r")])
    out.wrap("profile_cov", profile_cov, [(0, "r"), (1, "r")])
    out.wrap(
        "profile_modes",
        J_profile_mode * np.sqrt(evals)[None, :],
        [(0, "r"), (1, "mode")],
    )
    out.wrap("ell", ell, [(0, "ell")])
    out.wrap("bl", bl, [(0, "ell")])
    out.wrap("bl_sigma", bl_sigma, [(0, "ell")])
    out.wrap(
        "bl_modes",
        bl_mode * np.sqrt(evals)[None, :],
        [(0, "ell"), (1, "mode")],
    )
    out.wrap("cov_eigenvalues", evals, [(0, "mode")])
    out.wrap("cov_eigenvectors", evecs, [(0, "param"), (1, "mode")])
    out.wrap("profile_norm", profile_norm)
    out.wrap("lmax", lmax)
    out.wrap("n_modes", n_modes_actual)

    return out, cast(ndmap, model - off)
