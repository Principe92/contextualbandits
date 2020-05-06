import ctypes
from scipy.linalg.cython_blas cimport (
    ssyrk as tsyrk,
    sgemm as tgemm,
    sgemv as tgemv,
    ssymv as tsymv,
    ssymm as tsymm,
    ssyr as tsyr,
    scopy as tcopy,
    sdot as tdot,
    saxpy as taxpy,
    sscal as tscal,
    )
from scipy.linalg.cython_lapack cimport (
    sposv as tposv,
    spotri as tpotri,
    spotrf as tpotrf
    )

ctypedef float realtp
C_realtp = ctypes.c_float


include "linreg_untyped.pxi"
