from gate import Gate as Gate_S
from gate_SS import Gate as Gate_SS
from memory_layouts import *
import groq.api as g
import numpy as np



class Gate_IOP(Gate_S, Gate_SS):


	def __init__(self, pgm_pkg, mxm_permutation=True):
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation )
