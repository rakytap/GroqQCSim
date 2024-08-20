from split_gates import Split_gates
from memory_layouts import *
import groq.api as g

class Gate(Split_gates):
	def __init__(self, pgm_pkg, mxm_permutation=True):
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation)
	
	def gate_SL_program(self):
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_SL"
		
		with self.pgm_pkg.create_program_context(name) as pgm5:

			#g.reserve_tensor(pgm1, pgm2, State_input_real_mt)
				
			control_qbit = 9
			'''
			row_number_memory_map = 1 << (qbit_num-small_qbit_num_limit-4-4) #if the memory map is destributed over 2^4=16 slices and each 320 vector can store 4 independent addresses
			layout_LL_gate_map_selector = f"H1(W), A{row_number_memory_map}({small_qbit_num_limit*20}-{small_qbit_num_limit*20+row_number_memory_map-1}), S16(18-36)"

			dim_LL_gate_map_selector = 1 << (qbit_num-small_qbit_num_limit-4-4) # each 320 vector can store 8 independent addresses
			LL_gate_map_selector_mt = g.input_tensor(shape=(dim_LL_gate_map_selector*16,320,), dtype=g.uint8, name="LL_gate_map_selector", layout=layout_LL_gate_map_selector)
			'''
			

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
	
			
			
			#g.add_mem_constraints([LL_gate_map_selector_mt], [permute_maps_mt_west_shared], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 


			# Creates a shared tensor to reuse the memory allocation made by program pgm2

			# real and imag components of the broadcasted 00, 01, 10, 11 elements of the gate kernels (each element of a 320 vector contains either the 00,01,10 or 11 element)
			gate_kernels_broadcasted_real_shared_dict = {}
			gate_kernels_broadcasted_imag_shared_dict = {}
			gate_kernels_broadcasted_real_shared_copy_dict = {}
			gate_kernels_broadcasted_imag_shared_copy_dict = {}
			
			for label in gate_kernel_element_labels:

				key = label
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
				
			storage_req_real_diag     = g.tensor.create_storage_request( layout = layout_real_diag )
			storage_req_imag_diag     = g.tensor.create_storage_request( layout = layout_imag_diag )
			storage_req_real_off_diag = g.tensor.create_storage_request( layout = layout_real_off_diag )
			storage_req_imag_off_diag = g.tensor.create_storage_request( layout = layout_imag_off_diag )	

			###################################################################################
			# the transformed states are interated further from gate to gate, but for the first gate they are set to the initial state


			Psi_transformed_real_mt_list      = g.split( State_input_real_shared, 1 )
			Psi_transformed_imag_mt_list      = g.split( State_input_imag_shared, 1 )
			Psi_transformed_real_copy_mt_list = g.split( State_input_real_copy_shared, 1 )
			Psi_transformed_imag_copy_mt_list = g.split( State_input_imag_copy_shared, 1 )

			index_pair = 1 << (control_qbit - small_qbit_num_limit)


			mask_lower_bits = index_pair - 1 
			mask_upper_bits = 1 << ( qbit_num - control_qbit )
			mask_upper_bits = mask_upper_bits - 1
			mask_upper_bits = mask_upper_bits << (control_qbit - small_qbit_num_limit )


			control_qubit_state_1_indices = []

			half_memory_slices = memory_slices >> 1
			for memory_idx in range(half_memory_slices):

				lower_bits  = memory_idx & mask_lower_bits
				higher_bits = memory_idx & mask_upper_bits

				# index where the control qubit is in state 0
				state_index_control_qubit_0 = (higher_bits << 1) + lower_bits

				# index where the control qubit is in state 1
				state_index_control_qubit_1 = state_index_control_qubit_0 + index_pair

				control_qubit_state_1_indices.append( state_index_control_qubit_1 )

					

			states_real_mt      = g.concat([Psi_transformed_real_mt_list[index] for index in control_qubit_state_1_indices] )
			states_imag_mt      = g.concat([Psi_transformed_imag_mt_list[index] for index in control_qubit_state_1_indices] )
			states_real_copy_mt = g.concat([Psi_transformed_real_copy_mt_list[index] for index in control_qubit_state_1_indices] )
			states_imag_copy_mt = g.concat([Psi_transformed_imag_copy_mt_list[index] for index in control_qubit_state_1_indices] )
			########################################################################################x				
			with g.ResourceScope(name=f"prepare_state_pair_scope", is_buffered=True, time=0) as prepare_state_pair_scope :

				# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
				print('gatescope')   
				
				permuted_state_real_mt, permuted_state_imag_mt = self.permutation_of_S_qubits(states_real_mt, 
                                                                                                              states_imag_mt, 
                                                                                                                          states_real_copy_mt, 
                                                                                                                          states_imag_copy_mt)

				
				
			#################################################################################
			# component to perform the gate operation
				
			with g.ResourceScope(name=f"prepare_gate_kernel_scope", is_buffered=True, time=0) as prepare_gate_kernel_scope :
					
				# filter the elements in the vector according to the indices standig for output states |0> and |1> 

				# create_reusable_alu_requests
				alu_requests = []
				alu_requests.append(  g.tensor.create_alu_request( [0], 1 ) ) 
				alu_requests.append(  g.tensor.create_alu_request( [8], 1 ) ) 
				alu_requests.append(  g.tensor.create_alu_request( [5], 1 ) )
				alu_requests.append(  g.tensor.create_alu_request( [2], 1 ) )
					
				with g.ResourceScope(name=f"prepare_fileterd_gate_elements_scope", is_buffered=True, time=0) as prepare_fileterd_gate_elements_scope :

					state_1_broadcasted_st = state_1_broadcasted_shared.read( streams=g.SG4[1] )
					state_0_broadcasted_st = state_0_broadcasted_shared.read( streams=g.SG4[7] )

						
					# real and imaginary parts of the state concatenated into a unified tensor
					gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["00"].read( streams=g.SG4[6], time=0 )
					gate_00_imag_st = gate_kernels_broadcasted_imag_shared_dict["00"].read( streams=g.SG4[6], time=0 )
					gate_00_st 	= g.stack( [gate_00_real_st, gate_00_imag_st], dim=0 )

					#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
					gate_00_st = g.reinterpret( gate_00_st, g.uint32 )
					gate_00_st = g.bitwise_and( gate_00_st, state_0_broadcasted_st, alus=alu_requests[1], output_streams=g.SG4[4] )
					gate_00_st = g.reinterpret( gate_00_st, g.float32 )


					# real and imaginary parts of the state concatenated into a unified tensor
					gate_11_real_st = gate_kernels_broadcasted_real_shared_copy_dict["11"].read( streams=g.SG4[0])#, time=2 )
					gate_11_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict["11"].read( streams=g.SG4[0])#, time=2 )
					gate_11_st 	= g.stack( [gate_11_real_st, gate_11_imag_st], dim=0 )

					gate_11_st = g.reinterpret( gate_11_st, g.uint32 )
					gate_11_st = g.bitwise_and( gate_11_st, state_1_broadcasted_st, alus=alu_requests[0], output_streams=g.SG4[3] )
					gate_11_st = g.reinterpret( gate_11_st, g.float32 )

				
					# combine the filtered gate element vectors
					gate_diag_st = g.add(gate_00_st, gate_11_st, alus=alu_requests[2], output_streams=g.SG4[3])


					
					gate_diag_real_st, gate_diag_imag_st = g.split( gate_diag_st, num_splits=2, dim=0 )

					gate_diag_real_mt = gate_diag_real_st.write(name=f"gate_diag_real", storage_req=storage_req_real_diag)
					gate_diag_imag_mt = gate_diag_imag_st.write(name=f"gate_diag_imag", storage_req=storage_req_imag_diag)

				#with g.ResourceScope(name="prepare_gate_elements_output_state_1_scope", is_buffered=True, predecessors=[prepare_gate_elements_output_state_0_scope], time=None) as prepare_gate_elements_output_state_1_scope :

					state_1_broadcasted_st = state_1_broadcasted_shared.read( streams=g.SG4[1] )
					state_0_broadcasted_st = state_0_broadcasted_shared.read( streams=g.SG4[7] )
				
					# real and imaginary parts of the state concatenated into a unified tensor
					gate_01_real_st = gate_kernels_broadcasted_real_shared_dict["01"].read( streams=g.SG4[6])#, time=5 )
					gate_01_imag_st = gate_kernels_broadcasted_imag_shared_dict["01"].read( streams=g.SG4[6])#, time=5 )
					gate_01_st 	= g.stack( [gate_01_real_st, gate_01_imag_st], dim=0 )

					#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
					gate_01_st = g.reinterpret( gate_01_st, g.uint32 )
					gate_01_st = g.bitwise_and( gate_01_st, state_0_broadcasted_st, alus=alu_requests[1], output_streams=g.SG4[4] )
					gate_01_st = g.reinterpret( gate_01_st, g.float32 )

					# real and imaginary parts of the state concatenated into a unified tensor
					gate_10_real_st = gate_kernels_broadcasted_real_shared_copy_dict["10"].read( streams=g.SG4[0], time=7 )
					gate_10_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict["10"].read( streams=g.SG4[0], time=7 )
					gate_10_st 	= g.stack( [gate_10_real_st, gate_10_imag_st], dim=0 )

					gate_10_st = g.reinterpret( gate_10_st, g.uint32 )
					gate_10_st = g.bitwise_and( gate_10_st, state_1_broadcasted_st, alus=alu_requests[0], output_streams=g.SG4[3] )
					gate_10_st = g.reinterpret( gate_10_st, g.float32 )
					
					# combine the filtered gate element vectors
					gate_off_diag_st = g.add(gate_01_st, gate_10_st, alus=alu_requests[2], output_streams=g.SG4[3])
					
					gate_off_diag_st_list	= g.split( gate_off_diag_st, num_splits=2, dim=0 )
					gate_off_diag_real_mt 	= gate_off_diag_st_list[0].write(name=f"gate_off_diag_real", storage_req=storage_req_real_off_diag)
					gate_off_diag_imag_mt 	= gate_off_diag_st_list[1].write(name=f"gate_off_diag_imag", storage_req=storage_req_imag_off_diag)


			#################################################################################
				
			with g.ResourceScope(name=f"apply_gate_scope", is_buffered=True, predecessors=[prepare_state_pair_scope], time=None) as apply_gate_scope :

				
				State_input_imag_shared, State_input_imag_shared = self.VXM_S_round_trip(  states_real_mt, 
                                                                                                      states_imag_mt, 
                                                                                                      permuted_state_real_mt, 
                                                                                                      permuted_state_imag_mt, 
                                                                                                      gate_diag_real_mt, 
                                                                                                      gate_diag_imag_mt, 
                                                                                                      gate_off_diag_real_mt,
                                                                                                      gate_off_diag_imag_mt) 

				
					
					
			self.pgm_pkg.compile_program_context(pgm5)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
