from gate_S import Gate as Gate_S
from gate_SS import Gate as Gate_SS
from gate_SL import Gate as Gate_SL
from gate_S8 import Gate as Gate_S8
from gate_SH import Gate as Gate_SH
from memory_layouts import *
import groq.api as g
import numpy as np



class Gate_IOP(Gate_S, Gate_SS, Gate_SL, Gate_S8, Gate_SH):


	def __init__(self, pgm_pkg, mxm_permutation=True):
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation )
