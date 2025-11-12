from gate_counters import Gate_counters
from memory_layouts import *
import groq.api as g
import numpy as np

	
	

class Gate_8_base(Gate_counters):
	"""
	A class used to implement the gate operation on the target qubit 8 with no control qubits


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
		


	def two_complex_dot_product_8(self, s_r_0, s_i_0, s_r_1, s_i_1, g_r_0, g_i_0, g_r_1, g_i_1):
		"""
		Function used to perform the VXM round trip to get the transformed state
		
		
		Parameters
		----------
		s_r_0 : StreamTensor
			The real part of the input state vector, where the target qubit is in state 0
		s_i_0 : StreamTensor
			The imaginary part of the input state vector, where the target qubit is in state 0
		s_r_1 : StreamTensor
			The real part of the input state vector, where the target qubit is in state 1
		s_i_1 : StreamTensor
			The imaginary part of the input state vector, where the target qubit is in state 1
		g_r_0 : StreamTensor
			The real part of the first column of gate kernel orgnaized into first row -- second row -- first row -- second row -- .... sequence
		g_i_0 : StreamTensor
			The imaginary part of the first column of gate kernel orgnaized into first row -- second row -- first row -- second row -- .... sequence
		g_r_1 : StreamTensor
			The real part of the second column of gate kernel orgnaized into first row -- second row -- first row -- second row -- .... sequence
		g_i_1 : StreamTensor
			The imaginary part of the second column of gate kernel orgnaized into first row -- second row -- first row -- second row -- .... sequence  		
		
		Return
		------
		Returns with the stream tensors of the transformed state vector (real and imaginary parts)
		"""
		
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
		r_st = g.add(r0_st_m2, r1_st_m2, alus=[3], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[1], g.SG4_E[0]], direction=Direction.WESTWARD
		i_st = g.add(i0_st_m2, i1_st_m2, alus=[6], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[3], g.SG4_E[4]], direction=Direction.WESTWARD
		r_delay_st_p0 = g.vxm_identity(r_st, time=0, alus=[13], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[0]], direction=Direction.WESTWARD
		i_delay_st_p0 = g.vxm_identity(i_st, alus=[5], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[2]], direction=Direction.WESTWARD
		return r_st, i_st, r_delay_st_p0, i_delay_st_p0 #directions=[Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD]



	def read_out_gate_elements_8( self, element_num ):
	
		"""
		Function used to read out the the gate kernel elements.
		
		
		Parameters
		----------
		element_num : int
			The number of the required gate elements to perform the unitary transformation on all of the state vector elements
			
			
			
		Return
		------
		Returns with the stream tensors of the extracted gate elements			
			
		g_r_0 : StreamTensor
			The real part of the first column of gate kernel orgnaized into first row -- first row -- second row -- second row -- first row -- first row -- second row
			 -- second row -- .... sequence
		g_i_0 : StreamTensor
			The imaginary part of the first column of gate kernel orgnaized into first row -- first row -- second row -- second row -- first row -- first row -- 
			second row -- second row -- .... sequence
		g_r_1 : StreamTensor
			The real part of the second column of gate kernel orgnaized into first row -- first row -- second row -- second row -- first row -- first row -- s
			econd row -- second row -- .... sequence
		g_i_1 : StreamTensor
			The imaginary part of the second column of gate kernel orgnaized into first row -- first row -- second row -- second row -- first row -- first row -- 
			second row -- second row -- .... sequence  		
		
		"""
	
		gate_counter_gate_element_selector_00_shared = g.shared_memory_tensor( self.gate_counter_gate_element_selector_00_mt,  name="gate_counter_gate_element_selector_00_shared")
		gate_counter_gate_element_selector_10_shared = g.shared_memory_tensor( self.gate_counter_gate_element_selector_10_mt,  name="gate_counter_gate_element_selector_10_shared")
		gate_counter_gate_element_selector_01_shared = g.shared_memory_tensor( self.gate_counter_gate_element_selector_01_mt,  name="gate_counter_gate_element_selector_01_shared")
		gate_counter_gate_element_selector_11_shared = g.shared_memory_tensor( self.gate_counter_gate_element_selector_11_mt,  name="gate_counter_gate_element_selector_11_shared")
	
		gate_element_counter_first_column_list  = []
		gate_element_counter_second_column_list = []
		for idx in range(element_num//2):
			gate_element_counter_first_column_list.append( gate_counter_gate_element_selector_00_shared)
			gate_element_counter_first_column_list.append( gate_counter_gate_element_selector_10_shared)

			gate_element_counter_second_column_list.append( gate_counter_gate_element_selector_01_shared )
			gate_element_counter_second_column_list.append( gate_counter_gate_element_selector_11_shared )
					
					
		gate_element_counter_first_column_mt  = g.stack( gate_element_counter_first_column_list, dim=0 )
		gate_element_counter_first_column_mt  = gate_element_counter_first_column_mt.reshape( element_num, 320 )
		gate_element_counter_second_column_mt = g.stack( gate_element_counter_second_column_list, dim=0 )
		gate_element_counter_second_column_mt = gate_element_counter_second_column_mt.reshape( element_num, 320 )

			
		gate_element_counter_first_column_st  = gate_element_counter_first_column_mt.read( streams=g.SG1_W[17] )		
		gate_element_counter_second_column_st = gate_element_counter_second_column_mt.read( streams=g.SG1_W[18] )	
				
				
		gate_kernels_broadcasted_real_mt_8 = g.reinterpret( self.gate_kernels_broadcasted_real_mt, dtype=g.uint8 )
		gate_element_real_byte_list        = g.split( gate_kernels_broadcasted_real_mt_8, dim=1, num_splits=4 )
				
		gate_kernels_broadcasted_imag_mt_8 = g.reinterpret( self.gate_kernels_broadcasted_imag_mt, dtype=g.uint8 )
		gate_element_imag_byte_list        = g.split( gate_kernels_broadcasted_imag_mt_8, dim=1, num_splits=4 )

		gate_kernels_broadcasted_real_copy_mt_8 = g.reinterpret( self.gate_kernels_broadcasted_real_copy_mt, dtype=g.uint8 )
		gate_element_real_copy_byte_list        = g.split( gate_kernels_broadcasted_real_copy_mt_8, dim=1, num_splits=4 )

		gate_kernels_broadcasted_imag_copy_mt_8 = g.reinterpret( self.gate_kernels_broadcasted_imag_copy_mt, dtype=g.uint8 )
		gate_element_imag_copy_byte_list        = g.split( gate_kernels_broadcasted_imag_copy_mt_8, dim=1, num_splits=4 )
				
				
		gate_element_real_list = []
		gate_element_imag_list = []
		gate_element_real_copy_list = []
		gate_element_imag_copy_list = []
		for slice_idx in range(4):

			#### first column real part
			output_stream_offset = 7*4
			gate_element_real_byte_st = g.mem_gather(gate_element_real_byte_list[slice_idx], 
											gate_element_counter_first_column_st, 
											output_streams=[g.SG1_W[output_stream_offset+slice_idx]] )

			gate_element_real_byte_st = g.reshape( gate_element_real_byte_st, [element_num,1,320])
			gate_element_real_list.append( gate_element_real_byte_st )
					
			#### first column imag part					
			output_stream_offset = 3*4
			gate_element_imag_byte_st = g.mem_gather(gate_element_imag_byte_list[slice_idx], 
										gate_element_counter_first_column_st, 
										output_streams=[g.SG1[output_stream_offset+slice_idx]] )

			gate_element_imag_byte_st = g.reshape( gate_element_imag_byte_st, [element_num,1,320])
			gate_element_imag_list.append( gate_element_imag_byte_st )


			#### second column real part
			output_stream_offset = 5*4
			gate_element_real_copy_byte_st = g.mem_gather(gate_element_real_copy_byte_list[slice_idx], 
											gate_element_counter_second_column_st, 
											output_streams=[g.SG1_W[output_stream_offset+slice_idx]] )

			gate_element_real_copy_byte_st = g.reshape( gate_element_real_copy_byte_st, [element_num,1,320])
			gate_element_real_copy_list.append( gate_element_real_copy_byte_st )


			#### second column imag part
			output_stream_offset = 6*4
			gate_element_imag_copy_byte_st = g.mem_gather(gate_element_imag_copy_byte_list[slice_idx], 
											gate_element_counter_second_column_st, 
											output_streams=[g.SG1_W[output_stream_offset+slice_idx]] )

			gate_element_imag_copy_byte_st = g.reshape( gate_element_imag_copy_byte_st, [element_num,1,320])
			gate_element_imag_copy_list.append( gate_element_imag_copy_byte_st )
					


		gate_element_real_st      = g.concat( gate_element_real_list, dim=1 )
		gate_element_imag_st      = g.concat( gate_element_imag_list, dim=1 )
		gate_element_real_copy_st = g.concat( gate_element_real_copy_list, dim=1 )	
		gate_element_imag_copy_st = g.concat( gate_element_imag_copy_list, dim=1 )	
							
		gate_element_real_st      = g.reinterpret(gate_element_real_st, dtype=g.float32 )
		gate_element_imag_st      = g.reinterpret(gate_element_imag_st, dtype=g.float32 )				
		gate_element_real_copy_st = g.reinterpret(gate_element_real_copy_st, dtype=g.float32 )
		gate_element_imag_copy_st = g.reinterpret(gate_element_imag_copy_st, dtype=g.float32 )
		
		
		
		return gate_element_real_st, gate_element_imag_st, gate_element_real_copy_st, gate_element_imag_copy_st

	

	
