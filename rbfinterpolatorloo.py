
import numpy as np

from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import _rbfinterp as _rbfinterp

def _build_and_solve_system(y, d, smoothing, kernel, epsilon, powers):
    """Build and solve the RBF interpolation system of equations.

    Parameters
    ----------
    y : (P, N) float ndarray
        Data point coordinates.
    d : (P, S) float ndarray
        Data values at `y`.
    smoothing : (P,) float ndarray
        Smoothing parameter for each data point.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.

    Returns
    -------
    coeffs : (P + R, S) float ndarray
        Coefficients for each RBF and monomial.
    shift : (N,) float ndarray
        Domain shift used to create the polynomial matrix.
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.
    """
    
    lhs, rhs, shift, scale = _rbfinterp._build_system(
        y, d, smoothing, kernel, epsilon, powers
        )

    #_, _, coeffs, info = dgesv(lhs, rhs, overwrite_a=True, overwrite_b=True)
    sU, sS, sVt = np.linalg.svd(lhs)

    tmp1 = np.matmul(sU.T, rhs)
    tmp1 *= 1.0/ sS[:, None]
    coeffs = np.matmul(sVt.T, tmp1)

    return shift, scale, coeffs, sU, sS, sVt


class RBFLOOCVInterpolator(RBFInterpolator):

  def __init__(self, y, d,
               neighbors=None,
               smoothing=0.0,
               kernel="thin_plate_spline",
               epsilon=None,
               degree=None):
      y = np.asarray(y, dtype=float, order="C")
      if y.ndim != 2:
          raise ValueError("`y` must be a 2-dimensional array.")

      ny, ndim = y.shape

      d_dtype = complex if np.iscomplexobj(d) else float
      d = np.asarray(d, dtype=d_dtype, order="C")
      if d.shape[0] != ny:
          raise ValueError(
              f"Expected the first axis of `d` to have length {ny}."
              )

      d_shape = d.shape[1:]
      d = d.reshape((ny, -1))
      # If `d` is complex, convert it to a float array with twice as many
      # columns. Otherwise, the LHS matrix would need to be converted to
      # complex and take up 2x more memory than necessary.
      d = d.view(float)

      if np.isscalar(smoothing):
          smoothing = np.full(ny, smoothing, dtype=float)
      else:
          smoothing = np.asarray(smoothing, dtype=float, order="C")
          if smoothing.shape != (ny,):
              raise ValueError(
                  "Expected `smoothing` to be a scalar or have shape "
                  f"({ny},)."
                  )

      kernel = kernel.lower()
      if kernel not in _rbfinterp._AVAILABLE:
          raise ValueError(f"`kernel` must be one of {_AVAILABLE}.")

      if epsilon is None:
          if kernel in _rbfinterp._SCALE_INVARIANT:
              epsilon = 1.0
          else:
              raise ValueError(
                  "`epsilon` must be specified if `kernel` is not one of "
                  f"{_SCALE_INVARIANT}."
                  )
      else:
          epsilon = float(epsilon)

      min_degree = _rbfinterp._NAME_TO_MIN_DEGREE.get(kernel, -1)
      if degree is None:
          degree = max(min_degree, 0)
      else:
          degree = int(degree)
          if degree < -1:
              raise ValueError("`degree` must be at least -1.")
          elif degree < min_degree:
              warnings.warn(
                  f"`degree` should not be below {min_degree} when `kernel` "
                  f"is '{kernel}'. The interpolant may not be uniquely "
                  "solvable, and the smoothing parameter may have an "
                  "unintuitive effect.",
                  UserWarning
                  )

      if neighbors is None:
          nobs = ny
      else:
          # Make sure the number of nearest neighbors used for interpolation
          # does not exceed the number of observations.
          neighbors = int(min(neighbors, ny))
          nobs = neighbors

      powers = _rbfinterp._monomial_powers(ndim, degree)
      # The polynomial matrix must have full column rank in order for the
      # interpolant to be well-posed, which is not possible if there are
      # fewer observations than monomials.
      if powers.shape[0] > nobs:
          raise ValueError(
              f"At least {powers.shape[0]} data points are required when "
              f"`degree` is {degree} and the number of dimensions is {ndim}."
              )

      if neighbors is None:
          shift, scale, coeffs, sU, sS, sVt = _build_and_solve_system(
              y, d, smoothing, kernel, epsilon, powers
              )

          # Make these attributes private since they do not always exist.
          self._shift = shift
          self._scale = scale
          self._coeffs = coeffs

          self._sU = sU
          self._sS = sS
          self._sVt = sVt

      else:
          self._tree = KDTree(y)

      self.y = y
      self.d = d
      self.d_shape = d_shape
      self.d_dtype = d_dtype
      self.neighbors = neighbors
      self.smoothing = smoothing
      self.kernel = kernel
      self.epsilon = epsilon
      self.powers = powers


  def x__call__(self, x, d=None):
    """
    Allow different data to be provided to RBF.
    """
    
    if d is None:
      out = self(x)
    else:
      # check sizes
      npoints, npolyterms, ndim_rhs = self.y.shape[0], len(self.powers), d.shape[1]
      rhs = np.zeros((npoints+npolyterms, ndim_rhs))
      rhs[0:npoints, :] = d[:, :]
      tmp1 = np.matmul(self._sU.T, rhs)
      tmp1 *= 1.0/self._sS[:, None]
      _coeffs = np.matmul(self._sVt.T, tmp1)
      memory_budget = max(x.size + self.y.size + d.size, 1000000)

      if self.neighbors is None:

          out = self._chunk_evaluator(
              x,
              self.y,
              self._shift,
              self._scale,
              _coeffs,
              memory_budget=memory_budget)
      else:
        raise RuntimeError('Changing RHS (data matrix) is not supported with `self.neighbors`)

      out = out.view(self.d_dtype)
      d_shape = d.shape[1]
      nx, ndim = x.shape
      out = out.reshape((nx, d_shape))

    return out


  def get_loocv_coeff(self, index):
    # Compute x - (x[index]/invA[index,index]) * invA[:, index]

    npoint = self.y.shape[0]
    npoly_terms = len(self.powers)

    lhs = np.copy(self._coeffs)

    lhs_index = lhs[index, :]
    ident = np.zeros(self.y.shape[0] + npoly_terms)
    ident[index] = 1.0
    #print(self._sU.shape, self._sS.shape, self._sVt.shape)
    tmp1 = np.matmul(self._sU.T, ident)
    tmp1 *= 1.0/ self._sS
    invA_col = np.matmul(self._sVt.T, tmp1)

    #invA_col_rbfbasis = invA_col[:self.y.shape[0]]
    #lhs += -np.outer(invA_col_rbfbasis, lhs_index)/invA_col_rbfbasis[index]

    lhs += -np.outer(invA_col, lhs_index)/invA_col[index]

    #lhs[:, 0] += -invA_col * (lhs_index[0]/invA_col[index])

    return lhs


  def evaluate_loocv(self, x, index):
      """Evaluate the RBF-LOOCV interpolant at `x`.

      Parameters
      ----------
      x : (Q, N) array_like
          Evaluation point coordinates.
      index: int
          The index of the point to remove from the RBF.

      Returns
      -------
      (Q, ...) ndarray
          Values of the interpolant at `x`.
      """
      
      x = np.asarray(x, dtype=float, order="C")
      if x.ndim != 2:
          raise ValueError("`x` must be a 2-dimensional array.")

      nx, ndim = x.shape
      if ndim != self.y.shape[1]:
          raise ValueError("Expected the second axis of `x` to have length "
                           f"{self.y.shape[1]}.")

      # Our memory budget for storing RBF coefficients is
      # based on how many floats in memory we already occupy
      # If this number is below 1e6 we just use 1e6
      # This memory budget is used to decide how we chunk
      # the inputs
      memory_budget = max(x.size + self.y.size + self.d.size, 1000000)

      if self.neighbors is None:
          _coeffs = self.get_loocv_coeff(index)

          out = self._chunk_evaluator(
              x,
              self.y,
              self._shift,
              self._scale,
              _coeffs,
              memory_budget=memory_budget)
      else:
        raise RuntimeError('LOOCV is not supported with `self.neighbors`')

      out = out.view(self.d_dtype)
      out = out.reshape((nx, ) + self.d_shape)
      return out


  def evaluate_error(self, index):
      """Evaluate the interpolation error at `x[index]` using the RBF-LOOCV interpolant.

      Parameters
      ----------
      index: int
          The index of the point to remove from the RBF.

      Returns
      -------
      (Q, ...) ndarray
          Values of the interpolant at `x`.
      """

      nx = 1

      _x = np.zeros((1, self.y.shape[1]))
      _x[0, :] = self.y[index, :]
      _out = np.zeros((1, self.d.shape[1]))
      _out[0, :] = self.d[index, :]

      # Our memory budget for storing RBF coefficients is
      # based on how many floats in memory we already occupy
      # If this number is below 1e6 we just use 1e6
      # This memory budget is used to decide how we chunk
      # the inputs
      memory_budget = max(_x.size + self.y.size + self.d.size, 1000000)

      if self.neighbors is None:
          _coeffs = self.get_loocv_coeff(index)

          out = self._chunk_evaluator(
              _x,
              self.y,
              self._shift,
              self._scale,
              _coeffs,
              memory_budget=memory_budget)
      else:
        raise RuntimeError('LOOCV is not supported with `self.neighbors`')

      out = out.view(self.d_dtype)
      out = out.reshape((nx, ) + self.d_shape)
      return out


if __name__ == '__main__':
  index = 1

  ns = 20
  x_ = np.linspace(0.0, 4.0, ns)
  x0 = np.zeros((ns,1))
  x0[:,0] = x_[:]
  F = 1.1 + x_ * np.sin(np.pi * x_)
  xt = np.array( [[x_[index]]])
  rb = RBFInterpolator(x0, F, kernel="quintic")
  print('init', rb(xt))

  x_loo = np.zeros(ns-1)
  x_loo[0:index] = x_[0:index]
  x_loo[index:] = x_[index+1:]
  x0loo = np.zeros((ns-1,1))
  x0loo[:,0] = x_loo[:]
  Floo = 1.1 + x_loo * np.sin(np.pi * x_loo)
  xt = np.array( [[x_[index]]])
  rb = RBFInterpolator(x0loo, Floo, kernel="quintic")
  print('init-with-loo', rb(xt))
  print('loo error', F[index] - rb(xt))

  x0 = np.zeros((ns,1))
  x0[:,0] = x_[:]
  F = 1.1 + x_ * np.sin(np.pi * x_)

  print(x0[index])

  rbloo = RBFLOOCVInterpolator(x0, F, kernel="quintic")
  #print(F)
  #print(rbloo.d)
  print('loo ', rbloo(xt))
  xt_loo = rbloo.evaluate_loocv(xt, index)
  print('loo ', xt_loo)
  print('loo error', F[index] - rbloo.evaluate_error(index))


  F1 = 1.1 + x0 * np.sin(np.pi * x0)
  print('loo ', rbloo.x__call__(xt, d=F1))

  F1 = 1.1 + x0 * np.sin(np.pi * x0)
  F2 = np.zeros((F1.shape[0], 2))
  F2[:,0] = F1[:,0]
  F2[:,1] = F1[:,0]
  print('loo ', rbloo.x__call__(xt, d=F2))
