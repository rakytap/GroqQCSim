from gate_S import Gate as Gate_S
from gate_SS import Gate as Gate_SS
from gate_SL import Gate as Gate_SL
from gate_S8 import Gate as Gate_S8
from gate_SH import Gate as Gate_SH
from gate_L import Gate as Gate_L 
from gate_LL import Gate as Gate_LL
from gate_LS import Gate as Gate_LS
from gate_8 import Gate as Gate_8
from gate_8S import Gate as Gate_8S
from gate_L8 import Gate as Gate_L8
from memory_layouts import *
import groq.api as g
import numpy as np



class Gate_IOP(Gate_S, Gate_SS, Gate_SL, Gate_S8, Gate_SH, Gate_L, Gate_LL, Gate_LS, Gate_8, Gate_8S, Gate_L8):


	def __init__(self, pgm_pkg, mxm_permutation=True):
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation )
