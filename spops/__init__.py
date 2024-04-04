from spops.spmm import spmm
from spops.sddmm import sddmm
from spops.csr_transpose import csr_transpose
from spops.csr_add import csr_add
import os

def set_num_threads(num_threads):
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
