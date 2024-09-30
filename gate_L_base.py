from gate_counters import Gate_counters
from memory_layouts import *
import groq.api as g
import numpy as np

	
	

class Gate_L_base(Gate_counters):
	"""
	A class used to implement the gate operation on qubits S with no control qubits


	Attributes
	----------
    
	mxm_permutation : bool
		If true, MXM planes are used to generate permuted state pairs for the linear transfromation. If False, the permutor is used for the same task. 
		Permutor takes twice as much time as the MXM.

	"""

	def __init__(self, pgm_pkg, mxm_permutation=True):
		"""
		Parameters
		----------
		pgm_pkg : ProgramPackage
			The program package in which the gate operation program is generated
		mxm_permutation : bool
			If true, MXM planes are used to generate permuted state pairs for the linear transfromation. If False, the permutor is used for the same task. 
			Permutor takes twice as much time as the MXM.
		"""
        
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation )
		
		#self.target_qubit = 9

		self.transposer_request_to_qubit_state_readout  = g.tensor.create_transposer_request( [0], 1) # request transposer on the west side
		self.transposer_request_to_qubit_state_write    = g.tensor.create_transposer_request( [3], 1) # request transposer on the east side
		self.distributor_request_to_qubit_state_readout = g.tensor.create_distributor_request( [0], 1 )  # West side	
		self.distributor_request_to_qubit_state_write   = g.tensor.create_distributor_request( [6], 1 )  # East side	



	
	def two_complex_dot_product(self, s_r_0, s_i_0, s_r_1, s_i_1, g_r_0, g_i_0, g_r_1, g_i_1):
		
		s_r_0_st_m4 = s_r_0#.read(streams=g.SG4[4]) #direction=Direction.EASTWARD
		s_i_0_st_m4 = s_i_0#.read(streams=g.SG4[1]) #direction=Direction.EASTWARD
		s_r_1_st_m4 = s_r_1#.read(streams=g.SG4[5]) #direction=Direction.EASTWARD
		s_i_1_st_m4 = s_i_1#.read(streams=g.SG4[7]) #direction=Direction.EASTWARD
		g_r_0_st_m4 = g_r_0#.read(streams=g.SG4[7]) #direction=Direction.WESTWARD
		g_i_0_st_m4 = g_i_0#.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
		g_r_1_st_m4 = g_r_1#.read(streams=g.SG4[5]) #direction=Direction.WESTWARD
		g_i_1_st_m4 = g_i_1#.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
		rr0_st_m3 = g.mul(s_r_0_st_m4, g_r_0_st_m4, alus=[8], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[4], g.SG4_W[7]], direction=Direction.EASTWARD
		ii0_st_m3 = g.mul(s_i_0_st_m4, g_i_0_st_m4, alus=[0], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[1], g.SG4_W[3]], direction=Direction.EASTWARD
		ri0_st_m3 = g.mul(s_r_0_st_m4, g_i_0_st_m4, alus=[4], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[4], g.SG4_W[3]], direction=Direction.EASTWARD
		ir0_st_m3 = g.mul(s_i_0_st_m4, g_r_0_st_m4, alus=[12], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[1], g.SG4_W[7]], direction=Direction.EASTWARD
		r0_st_m2 = g.sub(rr0_st_m3, ii0_st_m3, alus=[14], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[6], g.SG4_E[1]], direction=Direction.EASTWARD
		i0_st_m2 = g.add(ri0_st_m3, ir0_st_m3, alus=[1], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[3], g.SG4_E[0]], direction=Direction.EASTWARD
		rr1_st_m3 = g.mul(s_r_1_st_m4, g_r_1_st_m4, alus=[7], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[5], g.SG4_W[5]], direction=Direction.WESTWARD
		ii1_st_m3 = g.mul(s_i_1_st_m4, g_i_1_st_m4, alus=[15], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[7], g.SG4_W[6]], direction=Direction.WESTWARD
		ri1_st_m3 = g.mul(s_r_1_st_m4, g_i_1_st_m4, alus=[10], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[5], g.SG4_W[6]], direction=Direction.WESTWARD
		ir1_st_m3 = g.mul(s_i_1_st_m4, g_r_1_st_m4, alus=[11], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[7], g.SG4_W[5]], direction=Direction.WESTWARD
		r1_st_m2 = g.sub(rr1_st_m3, ii1_st_m3, alus=[2], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[2], g.SG4_W[1]], direction=Direction.EASTWARD
		i1_st_m2 = g.add(ri1_st_m3, ir1_st_m3, alus=[9], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[6], g.SG4_W[5]], direction=Direction.EASTWARD
		r_st_m1 = g.add(r0_st_m2, r1_st_m2, alus=[3], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[1], g.SG4_E[0]], direction=Direction.WESTWARD
		i_st_m1 = g.add(i0_st_m2, i1_st_m2, alus=[6], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[3], g.SG4_E[4]], direction=Direction.WESTWARD
		r_delay_st_p0 = g.vxm_identity(r_st_m1, time=0, alus=[13], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[0]], direction=Direction.WESTWARD
		i_delay_st_p0 = g.vxm_identity(i_st_m1, alus=[5], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[2]], direction=Direction.WESTWARD
		return r_delay_st_p0, i_delay_st_p0 #directions=[Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD]
	


