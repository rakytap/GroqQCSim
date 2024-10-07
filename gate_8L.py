from gate_8_base import Gate_8_base
from memory_layouts import *
import groq.api as g
import numpy as np

	
	

class Gate(Gate_8_base):
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
		

		self.target_qubit = 8
		self.control_qubit = 9



	def two_complex_dot_product_8L(self, s_r_0, s_i_0, s_r_1, s_i_1, g_r_0, g_i_0, g_r_1, g_i_1):
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
		
		s_r_0_st_m4 = s_r_0.read(streams=g.SG4[4]) #direction=Direction.EASTWARD
		s_i_0_st_m4 = s_i_0.read(streams=g.SG4[1]) #direction=Direction.EASTWARD
		s_r_1_st_m4 = s_r_1.read(streams=g.SG4[5]) #direction=Direction.EASTWARD
		s_i_1_st_m4 = s_i_1.read(streams=g.SG4[7]) #direction=Direction.EASTWARD
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
	
	def gate_8L_program(self):
		"""
		Function implementing the L gate operation on the Groq chip
		"""
        
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_8L"
		
		with self.pgm_pkg.create_program_context(name) as pgm6:


			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			State_input_real_shared = g.shared_memory_tensor(self.State_input_real_mt, name="State_input_real_shared")
			State_input_imag_shared = g.shared_memory_tensor(self.State_input_imag_mt, name="State_input_imag_shared")
			State_input_real_copy_shared = g.shared_memory_tensor(self.State_input_real_copy_mt, name="State_input_real_copy_shared")
			State_input_imag_copy_shared = g.shared_memory_tensor(self.State_input_imag_copy_mt, name="State_input_imag_copy_shared")


			# the transformed states are interated further from gate to gate, but for the first gate they are set to the initial state
			Psi_transformed_real_mt = State_input_real_shared
			Psi_transformed_imag_mt = State_input_imag_shared
			Psi_transformed_real_copy_mt = State_input_real_copy_shared
			Psi_transformed_imag_copy_mt = State_input_imag_copy_shared



			with g.ResourceScope(name=f"apply_gate_scope", is_buffered=True, predecessors=[], time=0) as apply_gate_scope :
				control_1 = [ x^(1<<(self.control_qubit-8)) for x in range(memory_addresses) if x < x^(1<<(self.control_qubit-8)) ]

				pairs = []
				for x in range(memory_addresses):
					if x < x^1 and x in control_1:
						pairs.append([x,x^1])
                
				# print("=============================>", pairs)
				# print(len(inner_dims))
				# print(Psi_transformed_real_mt.physical_shape, Psi_transformed_real_mt.shape)
				#Psi_transformed_mt (memory_addresses, matrix_size)
				Psi_transformed_mt      = g.stack( [Psi_transformed_real_mt, Psi_transformed_imag_mt], dim=0).split(dim=1, num_splits=memory_addresses)
				Psi_transformed_copy_mt = g.stack( [Psi_transformed_real_copy_mt, Psi_transformed_imag_copy_mt], dim=0).split(dim=1, num_splits=memory_addresses)

				inner_dims      = [x.reshape(2, matrix_size) for x in Psi_transformed_mt]
				inner_dims_copy = [x.reshape(2, matrix_size) for x in Psi_transformed_copy_mt]
				#inner_dims (2, memory_addresses, matrix_size) -> [(2, matrix_size)]*memory_addresses 

				pairs_to_concat0_0 = []
				pairs_to_concat0_1 = []
				pairs_to_concat1_0 = []
				pairs_to_concat1_1 = []
				for i in pairs:
					pairs_to_concat0_0.append(inner_dims[i[0]])
					pairs_to_concat0_1.append(inner_dims[i[1]])
					pairs_to_concat1_0.append(inner_dims_copy[i[0]])
					pairs_to_concat1_1.append(inner_dims_copy[i[1]])

				#pairs_to_concat [(2, 1<<small_qbit_num_limit)]*(memory_addresses//2)
				pairs_to_concat0_0 = g.stack(pairs_to_concat0_0, dim=0)
				pairs_to_concat0_1 = g.stack(pairs_to_concat0_1, dim=0)
				pairs_to_concat1_0 = g.stack(pairs_to_concat1_0, dim=0)
				pairs_to_concat1_1 = g.stack(pairs_to_concat1_1, dim=0)
				#pairs_to_concat (memory_addresses//2, 2, matrix_size)

				pairs0 = g.stack([pairs_to_concat0_0, pairs_to_concat1_0], dim=0)
				pairs1 = g.stack([pairs_to_concat1_1, pairs_to_concat0_1], dim=0)
				
				pairs1 = [x.reshape(2, memory_addresses//4, matrix_size) for x in pairs1.split(num_splits=2, dim=2)]
				pairs0 = [x.reshape(2, memory_addresses//4, matrix_size) for x in pairs0.split(num_splits=2, dim=2)]
				#pairs (2, memory_addresses//4, 2, matrix_size) -> [(2, memory_addresses//4, matrix_size)]*2
				


				# readout gate elements	
				state_real_0_st = pairs0[0]
				element_num = state_real_0_st.shape[0]*state_real_0_st.shape[1]
						
				gate_first_column_real_st, gate_first_column_imag_st, gate_second_column_real_st, gate_second_column_imag_st = \
							self.read_out_gate_elements_8( element_num )
	 	
				
				new_shape = (2, memory_addresses//4, 1, matrix_size) if self.target_qubit==8 else (2, memory_addresses//8, 2, matrix_size)
				pairs0[0] = pairs0[0].reshape(*new_shape).transpose(1, 0, 2, 3)
				pairs0[1] = pairs0[1].reshape(*new_shape).transpose(1, 0, 2, 3)
				pairs1[0] = pairs1[0].reshape(*new_shape).transpose(1, 0, 2, 3)
				pairs1[1] = pairs1[1].reshape(*new_shape).transpose(1, 0, 2, 3)
                            
				pairs0[0], pairs0[1], pairs1[0], pairs1[1] = self.two_complex_dot_product_8L( pairs0[0],
														pairs0[1], 
														pairs1[0], 
														pairs1[1], 
														gate_first_column_real_st, 
														gate_first_column_imag_st, 
														gate_second_column_real_st, 
														gate_second_column_imag_st)



				pairs_to_store0 = g.stack([pairs_to_concat0_0, pairs_to_concat0_1], dim=0)
				pairs_to_store1 = g.stack([pairs_to_concat1_0, pairs_to_concat1_1], dim=0)
				#pairs_to_store (2, memory_addresses//2, 2, matrix_size)

				pairs0_0, pairs0_1 = g.split(pairs_to_store0, num_splits=2, dim=2)
				pairs1_0, pairs1_1 = g.split(pairs_to_store1, num_splits=2, dim=2)
				#pairs (2, memory_addresses//2, 1, matrix_size)
				
				pairs0_0 = pairs0_0.reshape(*new_shape).transpose(1, 0, 2, 3)
				pairs0_1 = pairs0_1.reshape(*new_shape).transpose(1, 0, 2, 3)
				pairs1_0 = pairs1_0.reshape(*new_shape).transpose(1, 0, 2, 3)
				pairs1_1 = pairs1_1.reshape(*new_shape).transpose(1, 0, 2, 3)

					#for target_qbit >= 9, bank alternation works by utilizing the VXM identity delayed output as it passes through even ALUs (compute graph has depth of 4 ALUs)
					#for target_qbit == 8, original is even/odd bank, copy is odd/even bank  
				pairs1, pairs0 = pairs0, pairs1
					

				pairs0[0].write(storage_req=pairs0_0.storage_request, name="pairs0_0")
				pairs0[1].write(storage_req=pairs0_1.storage_request, name="pairs0_1")
				pairs1[0].write(storage_req=pairs1_0.storage_request, name="pairs1_0")
				pairs1[1].write(storage_req=pairs1_1.storage_request, name="pairs1_1")

				'''
				Psi_transformed_real_mt_output = g.constant_tensor(
									Psi_transformed_real_mt.shape, 
									g.float32, 
									storage_req=Psi_transformed_real_mt.storage_request, 
									name=f"Psi_transformed_real_{gate_idx}", 
									initialized=False)

				Psi_transformed_imag_mt_output = g.constant_tensor(
					Psi_transformed_imag_mt.shape, g.float32, storage_req=Psi_transformed_imag_mt.storage_request, name=f"Psi_transformed_imag_{gate_idx}", initialized=False)

				Psi_transformed_real_mt_output.set_program_output()
				Psi_transformed_imag_mt_output.set_program_output()
				gr0.name = "gr0"
				gr1.name = "gr1"
				gi0.name = "gi0"
				gi1.name = "gi1"
				gr0.set_program_output()
				gr1.set_program_output()
				gi0.set_program_output()
				gi1.set_program_output()
				'''
				
			with g.ResourceScope(name=f"increase_gate_counters_scope", is_buffered=True, predecessors=[apply_gate_scope], time=None) as increase_gate_counters_scope :
			
				self.increment_gate_counters()	

				
			self.pgm_pkg.compile_program_context(pgm6)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
