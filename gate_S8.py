from gate_S_base import Gate_S_base
from memory_layouts import *
import groq.api as g
import numpy as np

	
	

class Gate(Gate_S_base):
	"""
	A class used to implement the gate operation on qubits S with control qubit 8


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
		
		

	
		
	
				
	
	def gate_S8_program(self):
		"""
		Function implementing the S8 gate operation on the Groq chip
		"""
        
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_S8"
		
		with self.pgm_pkg.create_program_context(name) as pgm5:


			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			State_input_real_shared      = g.shared_memory_tensor(self.State_input_real_mt, name="State_input_real_shared")
			State_input_imag_shared      = g.shared_memory_tensor(self.State_input_imag_mt, name="State_input_imag_shared")
			State_input_real_copy_shared = g.shared_memory_tensor(self.State_input_real_copy_mt, name="State_input_real_copy_shared")
			State_input_imag_copy_shared = g.shared_memory_tensor(self.State_input_imag_copy_mt, name="State_input_imag_copy_shared")

			
			

			# state_1_mt: if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
			# state_0_mt: if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			state_1_broadcasted_shared = g.shared_memory_tensor(self.state_1_broadcasted_mt, name="state_1_broadcasted_shared")
			state_0_broadcasted_shared = g.shared_memory_tensor(self.state_0_broadcasted_mt, name="state_0_broadcasted_shared")


			#state_1_broadcasted_list   = g.split(state_1_broadcasted_shared, dim=0, num_splits=gate_count)
			#state_0_broadcasted_list   = g.split(state_0_broadcasted_shared, dim=0, num_splits=gate_count)

			
			# Creates a shared tensor to reuse the memory allocation made by program pgm2

			# real and imag components of the broadcasted 00, 01, 10, 11 elements of the gate kernels (each element of a 320 vector contains either the 00,01,10 or 11 element)
			gate_kernels_broadcasted_real_shared_dict = {}
			gate_kernels_broadcasted_imag_shared_dict = {}
			gate_kernels_broadcasted_real_shared_copy_dict = {}
			gate_kernels_broadcasted_imag_shared_copy_dict = {}
			
			for label in gate_kernel_element_labels:

				key = '0_' + label
				gate_kernel_broadcasted_real_mt = self.gate_kernels_broadcasted_real_dict[key]
				gate_kernel_broadcasted_imag_mt = self.gate_kernels_broadcasted_imag_dict[key]

				gate_kernels_broadcasted_real_shared = g.shared_memory_tensor(gate_kernel_broadcasted_real_mt, name=gate_kernel_broadcasted_real_mt.name+"_shared")
				gate_kernels_broadcasted_imag_shared = g.shared_memory_tensor(gate_kernel_broadcasted_imag_mt, name=gate_kernel_broadcasted_imag_mt.name+"_shared")

				gate_kernels_broadcasted_real_shared_dict[key] = gate_kernels_broadcasted_real_shared
				gate_kernels_broadcasted_imag_shared_dict[key] = gate_kernels_broadcasted_imag_shared
					
				gate_kernel_broadcasted_real_copy_mt = self.gate_kernels_broadcasted_real_copy_dict[key]
				gate_kernel_broadcasted_imag_copy_mt = self.gate_kernels_broadcasted_imag_copy_dict[key]

				gate_kernels_broadcasted_real_shared_copy = g.shared_memory_tensor(gate_kernel_broadcasted_real_copy_mt, name=gate_kernel_broadcasted_real_copy_mt.name+"_shared")
				gate_kernels_broadcasted_imag_shared_copy = g.shared_memory_tensor(gate_kernel_broadcasted_imag_copy_mt, name=gate_kernel_broadcasted_imag_copy_mt.name+"_shared")

				gate_kernels_broadcasted_real_shared_copy_dict[key] = gate_kernels_broadcasted_real_shared_copy
				gate_kernels_broadcasted_imag_shared_copy_dict[key] = gate_kernels_broadcasted_imag_shared_copy				

			
			address_offset = 4*gate_count

			layout_real_diag = layout_gate_kernels_real + f", A1({address_offset})"
			layout_imag_diag = layout_gate_kernels_imag + f", A1({address_offset})"		
			layout_real_off_diag = layout_gate_kernels_real_copy + f", A1({address_offset})"
			layout_imag_off_diag = layout_gate_kernels_imag_copy + f", A1({address_offset})"	
				
			storage_req_real_diag  = g.tensor.create_storage_request( layout = layout_real_diag )
			storage_req_imag_diag  = g.tensor.create_storage_request( layout = layout_imag_diag )
			storage_req_real_off_diag = g.tensor.create_storage_request( layout = layout_real_off_diag )
			storage_req_imag_off_diag = g.tensor.create_storage_request( layout = layout_imag_off_diag )
		

					
							
			
				
			with g.ResourceScope(name=f"prepare_state_pair_scope", is_buffered=True, time=0) as prepare_state_pair_scope :

				# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
				print('gatescope')   
					
				if ( self.mxm_permutation ): # if MXM is used to generate qubit pairs
					
					permuted_state_real_mt, permuted_state_imag_mt = self.MXM_permutation_of_S_qubits(State_input_real_shared, 
                                                                                                                          State_input_imag_shared, 
                                                                                                                          State_input_real_copy_shared, 
                                                                                                                          State_input_imag_copy_shared)

					# we can only start the prepare_gate_kernel_scope after the weights have been install because of stream limitations
					tm_pkg = 25

				else: # if permutor is used to generate qubit pairs

					permuted_state_real_mt, permuted_state_imag_mt = self.permutation_of_S_qubits(State_input_real_shared, 
                                                                                                                          State_input_imag_shared, 
                                                                                                                          State_input_real_copy_shared, 
                                                                                                                          State_input_imag_copy_shared)

					tm_pkg = 0


				
			#################################################################################
			# component to perform the gate operation
				
			with g.ResourceScope(name=f"prepare_gate_kernel_scope", is_buffered=True, time=tm_pkg) as prepare_gate_kernel_scope :

		
				# combine the rows of the gate kernel to be used in the transformation on target qubits < 8
					
					
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 
					
				with g.ResourceScope(name=f"prepare_filtered_gate_elements_scope", is_buffered=True,  time=0) as prepare_filtered_gate_elements_scope :

					gate_counter_gate_filtering_selector_shared = g.shared_memory_tensor(self.gate_counter_gate_filtering_selector_mt,   \
												   										name="gate_counter_gate_filtering_selector_shared") 

					# gather streams to filter and mix gate elements for the unitary transformation
					state_1_broadcasted_st, state_0_broadcasted_st = self.gather_gate_filter_tensors()

					#state_1_broadcasted_st = state_1_broadcasted_list[0].read( streams=g.SG4[2] )
					#state_0_broadcasted_st = state_0_broadcasted_list[0].read( streams=g.SG4[0] )

					gate_counter_gate_element_selector_00_shared = g.shared_memory_tensor( \
									self.gate_counter_gate_element_selector_00_mt, \
									name="gate_counter_gate_element_selector_00_shared") 

					gate_counter_gate_element_selector_01_shared = g.shared_memory_tensor( \
									self.gate_counter_gate_element_selector_01_mt, \
									name="gate_counter_gate_element_selector_01_shared") 


					gate_counter_gate_element_selector_10_shared = g.shared_memory_tensor( \
									self.gate_counter_gate_element_selector_10_mt, \
									name="gate_counter_gate_element_selector_10_shared") 

					gate_counter_gate_element_selector_11_shared = g.shared_memory_tensor( \
									self.gate_counter_gate_element_selector_11_mt, \
									name="gate_counter_gate_element_selector_11_shared") 
					
					
					# real and imaginary parts of the gate element concatenated into a unified tensor
					gate_00_st = self.gather_gate_elements(gate_counter_gate_element_selector_00_shared, \
															g.SG1_W[30], \
															self.gate_kernels_broadcasted_real_mt, \
															self.gate_kernels_broadcasted_imag_mt, \
															output_stream_offset = 4, \
															time = 0)
															
					# real and imaginary parts of the gate element concatenated into a unified tensor
					gate_11_st = self.gather_gate_elements(gate_counter_gate_element_selector_11_shared, \
															g.SG1_W[31], \
															self.gate_kernels_broadcasted_real_copy_mt, \
															self.gate_kernels_broadcasted_imag_copy_mt, \
															output_stream_offset = 12 )

															
					gate_00_st = g.reinterpret( gate_00_st, g.uint32 )
					gate_00_st = g.bitwise_and( gate_00_st, state_0_broadcasted_st, alus=[12], output_streams=g.SG4[0] )
					gate_00_st = g.reinterpret( gate_00_st, g.float32 )



					gate_11_st = g.reinterpret( gate_11_st, g.uint32 )
					gate_11_st = g.bitwise_and( gate_11_st, state_1_broadcasted_st, alus=[4], output_streams=g.SG4[2] )
					gate_11_st = g.reinterpret( gate_11_st, g.float32 )

				
					# combine the filtered gate element vectors
					gate_diag_st = g.add(gate_00_st, gate_11_st, alus=[1], output_streams=g.SG4[3])
					
					
					gate_diag_st_list	= g.split( gate_diag_st, num_splits=2, dim=0 )
					gate_diag_real_mt 	= gate_diag_st_list[0].write(name=f"gate_diag_real", storage_req=storage_req_real_diag)#, program_output=True)
					gate_diag_imag_mt 	= gate_diag_st_list[1].write(name=f"gate_diag_imag", storage_req=storage_req_imag_diag)#, program_output=True)
					
					g.add_mem_constraints([self.gate_kernels_broadcasted_real_mt], [gate_diag_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
					g.add_mem_constraints([self.gate_kernels_broadcasted_imag_mt], [gate_diag_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)



					# gather streams to filter and mix gate elements for the unitary transformation
					state_1_broadcasted_st, state_0_broadcasted_st = self.gather_gate_filter_tensors()
					
					# real and imaginary parts of the gate element concatenated into a unified tensor
					gate_01_st = self.gather_gate_elements(gate_counter_gate_element_selector_01_shared, \
															g.SG1_W[30], \
															self.gate_kernels_broadcasted_real_mt, \
															self.gate_kernels_broadcasted_imag_mt, \
															output_stream_offset = 4 )
															
					# real and imaginary parts of the gate element concatenated into a unified tensor								
					gate_10_st = self.gather_gate_elements(gate_counter_gate_element_selector_10_shared, \
															g.SG1_W[31], \
															self.gate_kernels_broadcasted_real_copy_mt, \
															self.gate_kernels_broadcasted_imag_copy_mt, \
															output_stream_offset = 12, \
															time = 7 )
															
					gate_01_st = g.reinterpret( gate_01_st, g.uint32 )
					gate_01_st = g.bitwise_and( gate_01_st, state_0_broadcasted_st, alus=[14], output_streams=g.SG4[0] )
					gate_01_st = g.reinterpret( gate_01_st, g.float32 )


					gate_10_st = g.reinterpret( gate_10_st, g.uint32 )
					gate_10_st = g.bitwise_and( gate_10_st, state_1_broadcasted_st, alus=[6], output_streams=g.SG4[2] )
					gate_10_st = g.reinterpret( gate_10_st, g.float32 )
					
					# combine the filtered gate element vectors
					gate_off_diag_st = g.add(gate_01_st, gate_10_st, alus=[3], output_streams=g.SG4[3])
				
							
					
					gate_off_diag_st_list	= g.split( gate_off_diag_st, num_splits=2, dim=0 )
					gate_off_diag_real_mt 	= gate_off_diag_st_list[0].write(name=f"gate_off_diag_real", storage_req=storage_req_real_off_diag)
					gate_off_diag_imag_mt 	= gate_off_diag_st_list[1].write(name=f"gate_off_diag_imag", storage_req=storage_req_imag_off_diag)
		
					g.add_mem_constraints([self.gate_kernels_broadcasted_real_copy_mt], [gate_off_diag_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
					g.add_mem_constraints([self.gate_kernels_broadcasted_imag_copy_mt], [gate_off_diag_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				
				#################################################################################
			
			with g.ResourceScope(name=f"apply_gate_scope", is_buffered=True, predecessors=[prepare_state_pair_scope], time=None) as apply_gate_scope :
				# the number of inner split over which the state is distributed
				inner_splits = State_input_real_shared.shape[0]

				# define state with 320 vectors instead of 256 inner splits
				state_real_mt_320	 = g.constant_tensor((inner_splits, 320), dtype=g.float32, storage_req=State_input_real_shared.storage_request, initialized=False)		
				state_imag_mt_320	 = g.constant_tensor((inner_splits, 320), dtype=g.float32, storage_req=State_input_imag_shared.storage_request, initialized=False)
				state_real_copy_mt_320 = g.constant_tensor((inner_splits, 320), dtype=g.float32, storage_req=permuted_state_real_mt.storage_request, initialized=False)
				state_imag_copy_mt_320 = g.constant_tensor((inner_splits, 320), dtype=g.float32, storage_req=permuted_state_imag_mt.storage_request, initialized=False)		


				state_real_st 			  = state_real_mt_320.read( streams=[g.SG4[0]], time=0 )	
				state_imag_st 			  = state_imag_mt_320.read( streams=[g.SG4[3]] )
				state_permuted_real_st	    = state_real_copy_mt_320.read( streams=[g.SG4_E[6]] )
				state_permuted_imag_st 		  = state_imag_copy_mt_320.read( streams=[g.SG4[4]] )

				Psi_transformed_real_st, Psi_transformed_imag_st = self.VXM_S_round_trip_st(state_real_st, 
																				  state_imag_st, 
																				  state_permuted_real_st, 
																				  state_permuted_imag_st, 
																				  gate_diag_real_mt, 
																				  gate_diag_imag_mt, 
																				  gate_off_diag_real_mt, 
																				  gate_off_diag_imag_mt)

				# We have to select the adresses where qbit 8 is 1, we push the whole vector through the VXM 
				# because it will take the same amount of time as if we did it to half of it, because the 8th
				# qbit is 1 on every other adress so on the whole thing is on the same bank, but we only want
				# to overwrite the adresses where qbit 8 is 1
				Psi_transformed_real_st_list = g.split( Psi_transformed_real_st, 1 )
				Psi_transformed_imag_st_list = g.split( Psi_transformed_imag_st, 1 )
				State_input_real_shared_list = g.split( State_input_real_shared, 1)
				State_input_imag_shared_list = g.split( State_input_imag_shared, 1)
				permuted_state_real_mt_list = g.split( permuted_state_real_mt, 1 )
				permuted_state_imag_mt_list = g.split( permuted_state_imag_mt, 1 )
				Psi_transformed_real_st_half_list = []
				Psi_transformed_imag_st_half_list = []
				State_input_real_shared_half_list = []
				State_input_imag_shared_half_list = []
				permuted_state_real_mt_half_list = []
				permuted_state_imag_mt_half_list = []

				index_pair = 1

				half_memory_slices = memory_slices >> 1

				stagger_time = 20

				if (stagger_time % 2 ) > 0:
					raise('Stegger time should be a multiple of 2')

				stagger_cycles = half_memory_slices // stagger_time
				if ( memory_slices % stagger_time ) > 0:
					stagger_cycles = stagger_cycles + 1 

				mask_lower_bits = index_pair - 1 
				mask_upper_bits = 1 << ( qbit_num - 8 )
				mask_upper_bits = mask_upper_bits - 1
				mask_upper_bits = mask_upper_bits << (8 - small_qbit_num_limit )

			
				address_offset = 4*gate_count


				index = 0
				for outer_idx in range(stagger_cycles):
					index_loc = index
					# gathering the elements where the control qbit is 1
					for stagger_idx in range(stagger_time):
						lower_bits = index_loc & mask_lower_bits
						higher_bits = index_loc & mask_upper_bits

						state_index_control_qubit_0 = (higher_bits << 1) + lower_bits
						state_index_control_qubit_1 = state_index_control_qubit_0 + index_pair

						Psi_transformed_real_st_half_list.append(Psi_transformed_real_st_list[state_index_control_qubit_1])
						Psi_transformed_imag_st_half_list.append(Psi_transformed_imag_st_list[state_index_control_qubit_1])
						State_input_real_shared_half_list.append(State_input_real_shared_list[state_index_control_qubit_1])
						State_input_imag_shared_half_list.append(State_input_imag_shared_list[state_index_control_qubit_1])
						permuted_state_real_mt_half_list.append(permuted_state_real_mt_list[state_index_control_qubit_1])
						permuted_state_imag_mt_half_list.append(permuted_state_imag_mt_list[state_index_control_qubit_1])

						index_loc += 1

						if index_loc >= half_memory_slices:
							break
				
					index = index_loc
					if index >= half_memory_slices:
						break

				Psi_transformed_real_half_st = g.concat( Psi_transformed_real_st_half_list )
				Psi_transformed_imag_half_st = g.concat( Psi_transformed_imag_st_half_list )
				State_input_real_shared_half = g.concat( State_input_real_shared_half_list )
				State_input_imag_shared_half = g.concat( State_input_imag_shared_half_list )
				permuted_state_real_half_mt = g.concat( permuted_state_real_mt_half_list )
				permuted_state_imag_half_mt = g.concat( permuted_state_imag_mt_half_list )


				Psi_transformed_real_half_st.write(name=f"Psi_transformed_real_half", storage_req=State_input_real_shared_half.storage_request)
				Psi_transformed_real_half_st.write(name=f"Psi_transformed_real_half_copy", storage_req=permuted_state_real_half_mt.storage_request)

				Psi_transformed_imag_half_st.write(name=f"Psi_transformed_imag_half", storage_req=State_input_imag_shared_half.storage_request)
				Psi_transformed_imag_half_st.write(name=f"Psi_transformed_imag_half_copy", storage_req=permuted_state_imag_half_mt.storage_request)

				State_input_real_shared = g.constant_tensor(State_input_real_shared.shape, g.float32, storage_req=State_input_real_shared.storage_request, name="Psi_transformed_real", initialized=False)
				State_input_imag_shared = g.constant_tensor(State_input_imag_shared.shape, g.float32, storage_req=State_input_imag_shared.storage_request, name="Psi_transformed_imag", initialized=False)

				State_input_real_shared.set_program_output()
				State_input_imag_shared.set_program_output()


                                                                                                      
			                                                                           
			with g.ResourceScope(name=f"increase_gate_counters_scope", is_buffered=True, predecessors=[apply_gate_scope], time=None) as increase_gate_counters_scope :
			
				self.increment_gate_counters_for_S_gates()	

				

				
			self.pgm_pkg.compile_program_context(pgm5)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
