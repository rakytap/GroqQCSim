from gate_L_base import Gate_L_base
from memory_layouts import *
import groq.api as g
import numpy as np

	
	

class Gate(Gate_L_base):
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
	



	def write_tranformed_states_2_mem_L8( self, Psi_transformed_real_st, Psi_transformed_imag_st, target_qubit_L_write_map_st):
	
		"""
		Function used to scatter the transformed state vector into the memory. Every second calculated data are stored according to the control qubit state 1 of qubit 8
		Due to the bank alternation strategy, no better startegy was found.
		
		
		Parameters
		----------
		Psi_transformed_real_st : StreamTensor
			The real part of the transfromed state vector (both 0 and 1 states according to the first and second row of the gate kernel) 
		Psi_transformed_real_st : StreamTensor
			The imaginary part of the transfromed state vector (both 0 and 1 states according to the first and second row of the gate kernel) 
		target_qubit_L_write_map_st : StreamTensor
			The stream tensor for mem scatter operation
		
		"""	
	
		# define state with 320 vectors instead of 256 inner splits
		Psi_real_mt_320      = g.constant_tensor((memory_addresses, 320), dtype=g.float32, storage_req=self.State_input_real_mt.storage_request, initialized=False)		
		Psi_imag_mt_320      = g.constant_tensor((memory_addresses, 320), dtype=g.float32, storage_req=self.State_input_imag_mt.storage_request, initialized=False)
		Psi_real_copy_mt_320 = g.constant_tensor((memory_addresses, 320), dtype=g.float32, storage_req=self.State_input_real_copy_mt.storage_request, initialized=False)
		Psi_imag_copy_mt_320 = g.constant_tensor((memory_addresses, 320), dtype=g.float32, storage_req=self.State_input_imag_copy_mt.storage_request, initialized=False)

				
		# separating the tensors along the slice dimension (separating the bytes)
		Psi_real_mt_320_8       = g.reinterpret(Psi_real_mt_320, dtype=g.int8)
		Psi_imag_mt_320_8       = g.reinterpret(Psi_imag_mt_320, dtype=g.int8)			
		Psi_real_bytes_mt_list  = g.split(Psi_real_mt_320_8, dim=-2, num_splits=g.float32.size)
		Psi_imag_bytes_mt_list  = g.split(Psi_imag_mt_320_8, dim=-2, num_splits=g.float32.size)


		Psi_real_copy_mt_320_8       = g.reinterpret(Psi_real_copy_mt_320, dtype=g.int8)
		Psi_imag_copy_mt_320_8       = g.reinterpret(Psi_imag_copy_mt_320, dtype=g.int8)			
		Psi_real_bytes_copy_mt_list  = g.split(Psi_real_copy_mt_320_8, dim=-2, num_splits=g.float32.size)
		Psi_imag_bytes_copy_mt_list  = g.split(Psi_imag_copy_mt_320_8, dim=-2, num_splits=g.float32.size)
	
	
		# write the calculated states into the memory with mem scatter
		
		#choose every second data element in the streams to get control qubit state = 1 for qubit 8
		Psi_transformed_real_st_list        = g.split(Psi_transformed_real_st, dim=0, num_splits=memory_addresses )
		Psi_transformed_real_st_second_half = g.stack( Psi_transformed_real_st_list[1::2], dim=0 ) # data stored in bank0 corresponding to control qubit state 1	
		Psi_transformed_real_st_first_half  = g.stack( Psi_transformed_real_st_list[0::2], dim=0 )	# data stored in bank0 corresponding to control qubit state 0	
		
		Psi_transformed_imag_st_list        = g.split(Psi_transformed_imag_st, dim=0, num_splits=memory_addresses )
		Psi_transformed_imag_st_second_half = g.stack( Psi_transformed_imag_st_list[1::2], dim=0 ) # data stored in bank0 corresponding to control qubit state 1
		Psi_transformed_imag_st_first_half  = g.stack( Psi_transformed_imag_st_list[0::2], dim=0 )	# data stored in bank0 corresponding to control qubit state 0	
		
		target_qubit_L_write_map_st_list        = g.split(target_qubit_L_write_map_st, dim=0, num_splits=memory_addresses )
		target_qubit_L_write_map_st_second_half = g.stack( target_qubit_L_write_map_st_list[1::2], dim=0 ) # data stored in bank0 corresponding to control qubit state 1
		target_qubit_L_write_map_st_first_half  = g.stack( target_qubit_L_write_map_st_list[0::2], dim=0 )	# data stored in bank0 corresponding to control qubit state 0	
		

				
		
		Psi_transformed_real_mt_320_8       = g.reinterpret(Psi_transformed_real_st_second_half, dtype=g.int8)
		Psi_transformed_imag_mt_320_8       = g.reinterpret(Psi_transformed_imag_st_second_half, dtype=g.int8)			
		Psi_transformed_real_bytes_mt_list  = g.split(Psi_transformed_real_mt_320_8, dim=-2, num_splits=g.float32.size)
		Psi_transformed_imag_bytes_mt_list  = g.split(Psi_transformed_imag_mt_320_8, dim=-2, num_splits=g.float32.size)	

		Psi_transformed_real_mt_320_8_first_half       = g.reinterpret(Psi_transformed_real_st_first_half, dtype=g.int8)
		Psi_transformed_imag_mt_320_8_first_half       = g.reinterpret(Psi_transformed_imag_st_first_half, dtype=g.int8)			
		Psi_transformed_real_bytes_mt_list_first_half  = g.split(Psi_transformed_real_mt_320_8_first_half, dim=-2, num_splits=g.float32.size)
		Psi_transformed_imag_bytes_mt_list_first_half  = g.split(Psi_transformed_imag_mt_320_8_first_half, dim=-2, num_splits=g.float32.size)	
		
		# constant tensors to address the slices for temporary storage as unsaved data give timing errors during compilation
		dummy_space = g.constant_tensor( [memory_addresses//2, 320], 
						g.float32, 
						storage_req=self.dummy_space_storage_req,
						name=f"dummy_space_for_L8", 
						initialized=False)
						
		dummy_space_2 = g.constant_tensor( [memory_addresses//2, 320], 
						g.float32, 
						storage_req=self.dummy_space_2_storage_req,
						name=f"dummy_space_for_L8_2",
						initialized=False)
        
		dummy_space_8     = g.reinterpret(dummy_space, dtype=g.int8)
		dummy_space_2_8   = g.reinterpret(dummy_space_2, dtype=g.int8)			
		dummy_space_list  = g.split(dummy_space_8, dim=-2, num_splits=g.float32.size)
		dummy_space2_list = g.split(dummy_space_2_8, dim=-2, num_splits=g.float32.size)	
		
				
		for byte_idx in range(4):
					
			g.mem_scatter( Psi_transformed_real_bytes_mt_list[byte_idx], Psi_real_bytes_mt_list[byte_idx], target_qubit_L_write_map_st_second_half)
			g.mem_scatter( Psi_transformed_real_bytes_mt_list[byte_idx], Psi_real_bytes_copy_mt_list[byte_idx], target_qubit_L_write_map_st_second_half)
			g.mem_scatter( Psi_transformed_imag_bytes_mt_list[byte_idx], Psi_imag_bytes_mt_list[byte_idx], target_qubit_L_write_map_st_second_half)
			g.mem_scatter( Psi_transformed_imag_bytes_mt_list[byte_idx], Psi_imag_bytes_copy_mt_list[byte_idx], target_qubit_L_write_map_st_second_half)
			
			# dummy storage of unneeded data
			g.mem_scatter( Psi_transformed_real_bytes_mt_list_first_half[byte_idx], dummy_space_list[byte_idx], target_qubit_L_write_map_st_first_half)
			g.mem_scatter( Psi_transformed_imag_bytes_mt_list_first_half[byte_idx], dummy_space2_list[byte_idx], target_qubit_L_write_map_st_first_half)			
					
					
					

		#Psi_transformed_real_mt_output = g.constant_tensor( self.State_input_real_mt.shape, g.float32, storage_req=self.State_input_real_mt.storage_request, 
                #                                                                    name=f"Psi_transformed_real", initialized=False)
		#Psi_transformed_imag_mt_output = g.constant_tensor( self.State_input_imag_mt.shape, g.float32, storage_req=self.State_input_imag_mt.storage_request, 
                #                                                                    name=f"Psi_transformed_imag", initialized=False)

		#Psi_transformed_real_mt_output.set_program_output()
		#Psi_transformed_imag_mt_output.set_program_output()


	
		
	
	def gate_L8_program(self):
		"""
		Function implementing the L8 gate operation on the Groq chip
		"""
        
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_L8"
		
		with self.pgm_pkg.create_program_context(name) as pgm6:


			with g.ResourceScope(name=f"apply_gate_scope", is_buffered=True, predecessors=[], time=0) as apply_gate_scope :


				#index_pairs = [ [x, x^(1<<(self.target_qubit-8))] for x in range(memory_addresses) if x < x^(1<<(self.target_qubit-8)) ]

				read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st = self.select_target_qbit_read_map_west()
				
				# readout quantum state to be stransformed
				Psi_transformed_real_state_0_st, Psi_transformed_imag_state_0_st, Psi_transformed_real_state_1_st, Psi_transformed_imag_state_1_st  = \
							 self.read_out_state_2_transform( read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st)

					
				# readout gate elements	
				element_num = read_out_qubit_0_memory_tensor_st.shape[0]
						
				gate_element_real_st, gate_element_imag_st, gate_element_real_copy_st, gate_element_imag_copy_st = \
							self.read_out_gate_elements( element_num )	
						
				
				
				

				# calculate the VXM round trip                            
				Psi_transformed_real_st, Psi_transformed_imag_st = self.two_complex_dot_product( 
												Psi_transformed_real_state_0_st,#pairs0[0], 
												Psi_transformed_imag_state_0_st,#pairs0[1], 
												Psi_transformed_real_state_1_st,#pairs1[0], 
												Psi_transformed_imag_state_1_st,#pairs1[1], 
												gate_element_real_st, 
												gate_element_imag_st, 
												gate_element_real_copy_st, 
												gate_element_imag_copy_st)
												
				
				
				# export the calculated state into memory and set program output
				target_qubit_L_write_map_st = self.select_target_qbit_write_map_east()
				self.write_tranformed_states_2_mem_L8( Psi_transformed_real_st, Psi_transformed_imag_st, target_qubit_L_write_map_st )
				
				
			with g.ResourceScope(name=f"increase_gate_counters_scope", is_buffered=True, predecessors=[apply_gate_scope], time=None) as increase_gate_counters_scope :
			
				self.increment_gate_counters()	

				
			self.pgm_pkg.compile_program_context(pgm6)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
