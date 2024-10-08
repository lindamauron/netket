import jax
import jax.numpy as jnp

from netket.utils.struct import dataclass, field, Pytree

default_dtype = jnp.float64
r"""
To instantiate a Tableau for any order (not computed yet), one first needs to find the coefficients :
order = s
:math: `\beta_{s+1-i} = \int_{0}^1 \prod_{l=0 ,l \n eq i}^q \frac{u+l}{l-i} du, i=1,\dots,s`
:math: `\alpha_{i} = \int_{-1}^0 \prod_{l=0 ,l \n eq i}^q \frac{u+l}{l-i} du, i=0,\dots,s-1`
You can use the coeffs.mat file to generate the coefficients you need, if not yet provided. 

Then, instantiate as TableauABM(order=s, betas=jnp.array(betas), alphas=jnp.array(alphas), gamma=gamma, name=f"ABM{s}").

You can use the file `calculate_coefficients.nb` for this.
"""


# bashforths
bashforth_coefficients = {  # f{n-1},....,f_{n-s}
    1: jnp.array([1], default_dtype),
    2: jnp.array([3, -1], default_dtype) / 2,
    3: jnp.array([23, -16, 5], default_dtype) / 12,
    4: jnp.array([55, -59, 37, -9], default_dtype) / 24,
    5: jnp.array([1901, -2774, 2616, -1274, 251], default_dtype) / 720,
    6: jnp.array([4277, -7923, 9982, -7298, 2877, -475], default_dtype) / 1440,
    7: jnp.array(
        [198721, -447288, 705549, -688256, 407139, -134472, 19087], default_dtype
    )
    / 60480,
    8: jnp.array(
        [434241, -1152169, 2183877, -2664477, 2102243, -1041723, 295767, -36799],
        default_dtype,
    )
    / 120960,
}

# moulton
moulton_coefficients = {  # f_{n},...,f_{n-s+1}
    0: jnp.array([1], default_dtype),
    1: jnp.array([1, 1], default_dtype) / 2,
    2: jnp.array([5, 8, -1], default_dtype) / 12,
    3: jnp.array([9, 19, -5, 1], default_dtype) / 24,
    4: jnp.array([251, 646, -264, 106, -19], default_dtype) / 720,
    5: jnp.array([475, 1427, -798, 482, -173, 27], default_dtype) / 1440,
    6: jnp.array([19087, 65112, -46461, 37504, -20211, 6312, -863], default_dtype)
    / 60480,
    7: jnp.array(
        [36799, 139849, -121797, 123133, -88547, 41499, -11351, 1375], default_dtype
    )
    / 120960,
}

error_coefficients = {
    1: jnp.array([1 / 2], default_dtype),
    2: jnp.array([1 / 6], default_dtype),
    3: jnp.array([1 / 10], default_dtype),
    4: jnp.array([19 / 270], default_dtype),
    5: jnp.array([27 / 502], default_dtype),
    6: jnp.array([863 / 19950], default_dtype),
    7: jnp.array([1375 / 38174], default_dtype),
    8: jnp.array([33953 / 1103970], default_dtype),
}


@dataclass
class TableauABM(Pytree):
    r"""
    Class representing the tableau of an explicit Adams-Bashforth-Moulton method [1,2,3] of order :math:`s`.
    This scheme assembles an Adams-Bashforth integration of order :math:`s`, corrected by an Adams-Moulton integration of order :math:`s-1`.
    Explicitely, given the ODE :math:`dy/dt = F(t, y)`, the predictor is obtained as:

    .. math::
        y_{l+1}^{(p)} = y_{l} + dt*\sum_{j=0}^{s-1} \beta_j F_{l-j}

    where the index l denotes the time step t_l, with the partial derivatives

    .. math::
        F_l = F(t_l, y_t_l).

    This prediction can then be corrected using the Adams-Moulton method:
    .. math::
        y_{l+1} = y_{l} + dt*\alpha_0*f(t_{l+1}, y_{l+1}^{(p)}) + dt*\sum_{j=1}^{s-1} \alpha_j F_{l-j+1}.

    If :code:`self.is_adaptive`, the predictor solution is used to estimate the error

    ..math::
        y_{\mathrm{err}} = \gamma |y_{l} - y_{l}^{(p)}|.

    [1] J. Stoer and R. Bulirsch, Introduction to Numerical Analysis, Springer NY (2002).
    [2] J. C. Butcher, Numerical Methods for Ordinary Differential Equations, John Wiley & Sons Ltd, 2008
    [3] https://en.wikipedia.org/wiki/Linear_multistep_method
    """

    order: tuple[int, int]
    """The order of the tableau"""

    betas: jax.numpy.ndarray = field(repr=False)
    """The Adams-Bashforth coefficients of the tableau"""

    alphas: jax.numpy.ndarray = field(repr=False)
    """The Adams-Moulton coefficients of the tableau"""

    gamma: jax.numpy.ndarray = field(repr=False)
    """The error coefficients of the tableau"""

    name: str = field(pytree_node=False, default="ABMTableau")
    """The name of the tableau."""

    def __init__(self, order, *, alphas=None, betas=None, gamma: float = None):
        r"""
        Args:
            order: The order of the integration scheme.
                For an order `s`, the Bashforth coefficients \beta are of order `s`
                and the Moulton coefficients \alpha of order `s-1`.
            alphas: The Moulton coefficients.
            betas: The Bashforth coefficients
            gamma: The error coefficient
        """
        if alphas is None and betas is None:
            if order not in list(bashforth_coefficients.keys()):
                raise ValueError(
                    f"The coefficients of the tableau are only pre-computed up to order 8 <= {order}, please provide them."
                )

            else:
                alphas = moulton_coefficients[order - 1]
                betas = bashforth_coefficients[order]
                gamma = error_coefficients[order]

        else:
            alphas = jnp.asarray(alphas, dtype=default_dtype).reshape(-1)
            betas = jnp.asarray(betas, dtype=default_dtype).reshape(-1)
            gamma = jnp.asarray(gamma, dtype=default_dtype).reshape(-1)
        if not jnp.isclose(alphas.sum(), 1, atol=1e-10) or not jnp.isclose(
            alphas.sum(), 1, atol=1e-10
        ):
            raise ValueError(
                "Your coefficients are not correctly defined (should sum-up to 1)."
            )
        if gamma.ndim != 1:
            raise ValueError(
                "Your error coefficient `gamma` should be one-dimensional."
            )

        if order != alphas.shape[0] or order != betas.shape[0]:
            raise ValueError(
                f"Then length of both coefficient-arrays should be {order}, instead got {alphas.shape[0]} and {betas.shape[0]}."
            )

        self.order = order
        self.alphas = alphas
        self.betas = betas
        self.gamma = gamma
        self.name = f"ABM{order}"

    def __repr__(self):
        return self.name

    @property
    def is_explicit(self):
        """Boolean indication whether the integrator is explicit."""
        return True

    @property
    def is_adaptive(self):
        """Boolean indication whether the integrator can be adaptive."""
        return True

    @property
    def stages(self):
        """
        Number of stages (equal to the number of evaluations of the ode function)
        of the scheme.
        """
        return 2

    @property
    def error_order(self):
        """
        Returns the order of the embedded error estimate for a tableau
        supporting adaptive step size. Otherwise, None is returned.
        """
        return self.order + 2
