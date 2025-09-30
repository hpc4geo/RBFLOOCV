
import time
import numpy as np
from scipy.interpolate import RBFInterpolator


def gen_matrix(M, N):
    A = np.random.random((M,N))
    u,s,vt = np.linalg.svd(A, full_matrices=False)
    return A, u, s, vt


def split_matrix(A, col_index):
    As = np.zeros((A.shape[0],A.shape[1]-1))
    cnt = 0
    for j in range(A.shape[1]):
        if j != col_index:
            As[:, cnt] = A[:, j]
            cnt += 1

    u,s,vt = np.linalg.svd(As, full_matrices=False)
    return As, u, s, vt


def drop_index(u, s, vt, col_index):
    _vt = np.copy(vt)
    _vt[:, col_index] = 0
    print(u @ np.diag(s) @ _vt)


def svd_drop_col(u, s, vt, col_index):
    _alpha2 = np.diag(s) @ vt
    _alpha2[:, col_index] = 0

    _u, _s, _vt = np.linalg.svd(_alpha2, full_matrices=False)

    return _u, _s, _vt


def svd_rbf_loocv(r, params, u, alpha, col_index):
    """
    Computes the Leave-One-Out CV for a snapshot.

    Computes the interpolated snapshot at `params[col_index, :]`
    assuming that the snapshot at `A[:, col_index] = u @ np.diag(s) vt`
    has been been dropped.

    Notes: This implementation will perform an SVD a system of
    size nsnaps x nsnaps.

    Input:
        - r: scipy.interpolate.RBFInterpolator
        - param: numpy.ndarray, shape = (nsnaps, nsnaps)
        - u: numpy.ndarray, shape = (M, nsnaps)
        - s: numpy.ndarray, shape = (nsnaps, )
        - vt: numpy.ndarray, shape = (nsnaps, nsnaps)

    Output:
        - x_loo: numpy.ndarray, shape = (M, 1)
    """
    N = u.shape[1]

    _alpha2 = np.copy(alpha)
    _alpha2[:, col_index] = 0

    t_svd = time.perf_counter()
    _u2, _s2, _vt2 = np.linalg.svd(_alpha2, full_matrices=False)
    t_svd = time.perf_counter() - t_svd
    print('     svd(s vt)', ('%1.4e' % t_svd), '(sec)')

    #print('u _u2\n', u @ _u2) # compare the first N-1 columns with the reduced u

    _alpha3 = np.diag(_s2) @ _vt2
    _alpha3[N-1, :] = 0

    #_alpha4 = np.copy( _alpha3.T[:, :(N-1)]   )
    _alpha4 = np.copy( _alpha3.T  )


    _idx = np.zeros(params.shape[0], dtype=np.bool_)
    _idx[:] = True
    _idx[col_index] = False

    _params = params[ _idx, :]
    _alpha4 = _alpha4[_idx, :]

    #_params = np.copy(params)

    #print('_alpha\n', _alpha4)

    t_rb = time.perf_counter()
    rb_loo = RBFInterpolator(_params, _alpha4, kernel=r.kernel)
    t_rb = time.perf_counter() - t_rb
    print('     rbf(P)', ('%1.4e' % t_rb), '(sec)')

    test_param = np.zeros((1,params.shape[1]))
    test_param[0, :] = params[col_index, :]
    _weights = rb_loo(test_param)

    #print('_weights\n', _weights)
    #weights = np.zeros(u.shape[1])

    #slots = np.arange(u.shape[1])
    #slots = slots[_idx]
    #weights[slots] = _weights[:]
    #print(weights)
    t_mat = time.perf_counter()
    x_tmp = _u2 @ _weights.T
    x_loo = u @ x_tmp
    t_mat = time.perf_counter() - t_mat
    print('     mat-mat-mult', ('%1.4e' % t_mat), '(sec)')

    x_loo.shape = (x_loo.shape[0], 1)

    return x_loo


def example_svd():
  col2drop = 2

  Nsnaps = 4

  A, u, s, vt = gen_matrix(6, Nsnaps)
  print('A\n', A)
  print('s\n',s)

  As, us, ss, vts = split_matrix(A, col2drop)
  print('As\n',As)
  print('ss\n',ss)
  print('vts\n',vts)
  #drop_index(u, s, vt, col2drop)

  _u, _s, _vt = svd_drop_col(u, s, vt, col2drop)

  print('_s\n',_s)
  print('_vt\n',_vt)

  print('a_recon\n',u @ _u @np.diag(_s) @ _vt)


def example_rbf():

  view = False

  M = 300000
  Nsnaps = 400
  col2drop = 2



  print('== Generate and factor matrix ==')
  t_gen = time.perf_counter()
  A, u, s, vt = gen_matrix(M, Nsnaps)
  t_gen = time.perf_counter() - t_gen
  if view: print('A\n', A)
  print('SVD(A)', ('%1.4e' % t_gen), '(sec)')

  p1 = np.linspace(0.0, 1.0, Nsnaps)
  p2 = np.linspace(1.0, 2.0, Nsnaps)
  params = np.zeros((Nsnaps, 2))
  params[:, 0] = p1[:]
  params[:, 1] = p2[:]

  alpha = np.diag(s) @ vt
  _alpha = np.copy(alpha.T)

  print('== Original RBF ==')
  t_rbf = time.perf_counter()
  rb_all = RBFInterpolator(params, _alpha, kernel="linear")
  t_rbf = time.perf_counter() - t_rbf
  print('RBF(s vt)', ('%1.4e' % t_rbf), '(sec)')

  test_param = np.zeros((1,2))
  test_param[0, 0], test_param[0, 1] = p1[0], p2[0]
  weights = rb_all(test_param)
  if view: print(weights)
  if view: print(u @ weights.T)


  print('== Brute force ==')
  t_brute = time.perf_counter()
  As, us, ss, vts = split_matrix(A, col2drop)
  if view: print('As\n',As)
  if view: print('us\n', us)

  _idx = np.zeros(params.shape[0], dtype=np.bool_)
  _idx[:] = True
  _idx[col2drop] = False

  _params = params[_idx, :]

  alpha = np.diag(ss) @ vts
  _alpha = np.copy(alpha.T)

  rb_bruteforce = RBFInterpolator(_params, _alpha, kernel="linear")
  if view: print('_alpha\n', _alpha)

  test_param = np.zeros((1,2))
  test_param[0, 0], test_param[0, 1] = p1[col2drop], p2[col2drop]
  weights = rb_bruteforce(test_param)
  if view: print('weights\n', weights)

  x_loo = us @ weights.T
  t_brute = time.perf_counter() - t_brute

  if view: print('x_loo\n', x_loo, x_loo.shape)
  print('RBF-SVD-BRUTE(s vt)', ('%1.4e' % t_brute), '(sec)')


  print('== Efficient ==')
  t_fast = time.perf_counter()
  x_loo_e = svd_rbf_loocv(rb_all, params, u, np.diag(s) @ vt, col2drop)
  t_fast = time.perf_counter() - t_fast
  print('RBF-SVD-FAST(s vt)', ('%1.4e' % t_fast), '(sec)')

  if view: print('x_loo\n', x_loo_e,  x_loo.shape)

  diff = np.absolute(x_loo - x_loo_e)
  print('max diff', np.max(diff))


  print('cost brute force', ( '%1.4e' % (t_brute * Nsnaps)), '(sec)')
  print('cost fast       ', ( '%1.4e' % (t_fast * Nsnaps)), '(sec)')


if __name__ == '__main__':
  np.random.seed(0)

  #example_svd()
  example_rbf()
