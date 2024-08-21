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
			
			#########################################
			control_qubit_address_maps_SL_west_shared = 	g.shared_memory_tensor(self.control_qubit_address_maps_SL_west_mt, name="control_qubit_address_maps_SL_west_shared")		
			control_qubit_address_maps_SL_west_list = g.split( control_qubit_address_maps_SL_west_shared, 1 )
			
			control_qubit_address_maps_SL_east_shared_a = 	g.shared_memory_tensor(self.control_qubit_address_maps_SL_east_a_mt, name="control_qubit_address_maps_SL_east_shared_a")		
			control_qubit_address_maps_SL_east_list_a = g.split( control_qubit_address_maps_SL_east_shared_a, 1 )
			
			control_qubit_address_maps_SL_east_shared_b = 	g.shared_memory_tensor(self.control_qubit_address_maps_SL_east_b_mt, name="control_qubit_address_maps_SL_east_shared_b")		
			control_qubit_address_maps_SL_east_list_b = g.split( control_qubit_address_maps_SL_east_shared_b, 1 )

			
			
			#################################################################################
			with g.ResourceScope(name=f"prepare_state_pair_scope", is_buffered=True, time=0) as prepare_state_pair_scope :
				# byte-wise permutation of the state vector
				Psi_transformed_real_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=State_input_real_shared.storage_request, initialized=False)		
				Psi_transformed_imag_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=State_input_imag_shared.storage_request, initialized=False)					

				Psi_transformed_real_mt_320 = g.reinterpret(Psi_transformed_real_mt_320, dtype=g.int8)
				Psi_transformed_imag_mt_320 = g.reinterpret(Psi_transformed_imag_mt_320, dtype=g.int8)			
				# separating the tensors along the slice dimension (separating the bytes)
				Psi_transformed_real_mt_list       = g.split(Psi_transformed_real_mt_320, dim=-2, num_splits=g.float32.size)
				Psi_transformed_imag_mt_list       = g.split(Psi_transformed_imag_mt_320, dim=-2, num_splits=g.float32.size)			
			
			
				Psi_transformed_real_copy_mt_list = g.split(g.reinterpret(State_input_real_copy_shared, dtype=g.int8), dim=-2, num_splits=g.float32.size)
				Psi_transformed_imag_copy_mt_list = g.split(g.reinterpret(State_input_imag_copy_shared, dtype=g.int8), dim=-2, num_splits=g.float32.size)			
			
			
				# create reusable permutor requests
				permutor_requests = []
				permutor_requests.append( g.tensor.create_permutor_request( [0], 1 ) )
				permutor_requests.append( g.tensor.create_permutor_request( [1], 1 ) )			
			
				half_memory_slices = memory_slices >> 1
				control_qubit_address_maps_SL_west_tmp   = g.concat( control_qubit_address_maps_SL_west_list[0:half_memory_slices] )
				control_qubit_address_maps_SL_east_tmp_a = g.concat( control_qubit_address_maps_SL_east_list_a[0:half_memory_slices] )
				control_qubit_address_maps_SL_east_tmp_b = g.concat( control_qubit_address_maps_SL_east_list_b[0:half_memory_slices] )
			
			
				state_permute_map_selector_shared = g.shared_memory_tensor(self.state_permute_map_selector_mt, name="state_permute_map_selector_shared")		
				state_permute_map_selector_list   = g.split_vectors(input=state_permute_map_selector_shared, splits=[1]*gate_count)
			
	
				# otherwise the permutor slices are used
				permute_maps_mt_west_shared = g.shared_memory_tensor(self.permute_maps_mt_west, name="permute_maps_west_shared")
				permute_maps_mt_east_shared = g.shared_memory_tensor(self.permute_maps_mt_east, name="permute_maps_east_shared")			

				memory_addresses = control_qubit_address_maps_SL_west_tmp.shape[0]
				permutation_time_starts = [0, memory_addresses, 2*memory_addresses, 3*memory_addresses]
				for byte_idx in range(4):
					
					# Real part
					control_qubit_address_maps_SL_west_st = control_qubit_address_maps_SL_west_tmp.read( streams = g.SG1_E[31] )
					control_qubit_address_maps_SL_east_a_st = control_qubit_address_maps_SL_east_tmp_a.read( streams = g.SG1_W[31] )
				
					state_real_byte_st                 = g.mem_gather(Psi_transformed_real_mt_list[byte_idx], control_qubit_address_maps_SL_west_st, output_streams=[g.SG1_E[0]] )				
					state_permute_map_selector_east_st = state_permute_map_selector_list[0].read(streams=[g.SG1_E[24]])	
					permute_map_west_st                = g.mem_gather(permute_maps_mt_east_shared, state_permute_map_selector_east_st, output_streams=[g.SG1_E[24]])								
					permuted_state_real_st_8           = g.permute_inner(state_real_byte_st, permute_map_west_st, permutor_req=permutor_requests[1], input_streams=[g.SG1_E[0], g.SG1_E[24]], output_streams=g.SG1[4], time=permutation_time_starts[byte_idx] )
				

					g.mem_scatter(permuted_state_real_st_8, Psi_transformed_real_copy_mt_list[byte_idx], control_qubit_address_maps_SL_east_a_st)
				
					# Imag part
					control_qubit_address_maps_SL_east_b_st = control_qubit_address_maps_SL_east_tmp_b.read( streams = g.SG1_W[30] )				
				
					state_imag_byte_st                 = g.mem_gather(Psi_transformed_imag_mt_list[byte_idx], control_qubit_address_maps_SL_west_st, output_streams=[g.SG1_W[0]] )				
					state_permute_map_selector_east_st = state_permute_map_selector_list[0].read(streams=[g.SG1_W[24]])	
					permute_map_west_st                = g.mem_gather(permute_maps_mt_west_shared, state_permute_map_selector_east_st, output_streams=[g.SG1_W[24]])								
					permuted_state_imag_st_8           = g.permute_inner(state_imag_byte_st, permute_map_west_st, permutor_req=permutor_requests[0], input_streams=[g.SG1_W[0], g.SG1_E[24]], output_streams=g.SG1[4])#, time=permutation_time_starts[byte_idx]+0*memory_addresses )
				

					g.mem_scatter(permuted_state_imag_st_8, Psi_transformed_imag_copy_mt_list[byte_idx], control_qubit_address_maps_SL_east_b_st)
				
				
					

			
			
			
			#########################################
			
			
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

			permuted_state_real_mt, permuted_state_imag_mt = states_real_copy_mt, states_imag_copy_mt
				
			with g.ResourceScope(name=f"apply_gate_scope", is_buffered=True, predecessors=[prepare_state_pair_scope], time=None) as apply_gate_scope :

				
				alu_rq_0  = g.tensor.create_alu_request( alus=[0] )
				alu_rq_1  = g.tensor.create_alu_request( alus=[1] )
				alu_rq_4  = g.tensor.create_alu_request( alus=[4] )			
				alu_rq_5  = g.tensor.create_alu_request( alus=[5] )				
				alu_rq_8  = g.tensor.create_alu_request( alus=[8] )				
				alu_rq_9  = g.tensor.create_alu_request( alus=[9] )							
				alu_rq_11 = g.tensor.create_alu_request( alus=[11] )			
				alu_rq_12 = g.tensor.create_alu_request( alus=[12] )
				alu_rq_13 = g.tensor.create_alu_request( alus=[13] )	

				# calculate Psi_real * gate_diag_real				
				state_real_st 		       = states_real_mt.read( streams=[g.SG4[0]] )	
				#gate_diag_real_mt_256         = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_real_diag, name="gate_diag_real_mt_256", initialized=False)	
				#gate_diag_real_st 	       = gate_diag_real_mt_256.read( streams=[g.SG4_W[0]] )
				#state_real__gate_diag_real_st = g.mul( state_real_st, gate_diag_real_st, alus=alu_rq_12, output_streams=g.SG4_E[0], time=150 )
				gate_diag_real_mt.latch(alus=alu_rq_12, input_streams=g.SG4_W[0], time=0)
				state_real__gate_diag_real_st = state_real_st.mul(None, alus=alu_rq_12, output_streams=g.SG4_E[0], time=2)
				

				# calculate Psi_imag * gate_diag_imag
				state_imag_st 		       = states_imag_mt.read( streams=[g.SG4[3]] )
				#gate_diag_imag_mt_256         = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_imag_diag, name="gate_diag_imag_mt_256", initialized=False)	
				#gate_diag_imag_st 	       = gate_diag_imag_mt_256.read( streams=[g.SG4_W[3]] )
				#state_imag__gate_diag_imag_st = g.mul( state_imag_st, gate_diag_imag_st, alus=[4], output_streams=g.SG4_E[2] )	
				gate_diag_imag_mt.latch(alus=alu_rq_4, input_streams=g.SG4_W[3], time=0)				
				state_imag__gate_diag_imag_st = state_imag_st.mul(None, alus=alu_rq_4, output_streams=g.SG4_E[2])
				

				# calculate Psi_real * gate_diag_real - Psi_imag * gate_diag_imag
				state__gate_diag_real_st	= g.sub( state_real__gate_diag_real_st, state_imag__gate_diag_imag_st, alus=[2], output_streams=g.SG4_E[2] )
						
				

	
					
					
				# calculate Psi_imag * gate_diag_real
				#state_imag__gate_diag_real_st	= g.mul( state_imag_st, gate_diag_real_st, alus=alu_rq_1, output_streams=g.SG4_E[3] )	
				gate_diag_real_mt.latch(alus=alu_rq_1, input_streams=g.SG4_W[0], time=1)
				state_imag__gate_diag_real_st = state_imag_st.mul(None, alus=alu_rq_1, output_streams=g.SG4_E[3])		
						
				# calculate Psi_real * gate_diag_imag
				#state_real__gate_diag_imag_st	= g.mul( state_real_st, gate_diag_imag_st, alus=[0], output_streams=g.SG4_E[1] )				
				gate_diag_imag_mt.latch(alus=alu_rq_0, input_streams=g.SG4_W[3], time=1)
				state_real__gate_diag_imag_st	= state_real_st.mul( None, alus=alu_rq_0, output_streams=g.SG4_E[1] )
				

				# calculate Psi_real * gate_diag_imag + Psi_imag * gate_diag_real
				state__gate_diag_imag_st	= g.add( state_imag__gate_diag_real_st, state_real__gate_diag_imag_st, alus=[3], output_streams=g.SG4_W[1]  )
				
					
						
					
				# calculate Psi_permuted_real * gate_off_diag_real
				state_permuted_real_st     = permuted_state_real_mt.read( streams=[g.SG4_E[6]] )
				#gate_off_diag_real_mt_256 = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_real_off_diag, name="gate_off_diag_real_mt_256", initialized=False)					
				#gate_off_diag_real_st 	                    = gate_off_diag_real_mt_256.read( streams=[g.SG4_W[6]] )
				#state_permuted_real__gate_off_diag_real_st = g.mul( state_permuted_real_st, gate_off_diag_real_st, alus=[13], output_streams=g.SG4_E[6] )
				gate_off_diag_real_mt.latch(alus=alu_rq_13, input_streams=g.SG4_W[6], time=0)				
				state_permuted_real__gate_off_diag_real_st = state_permuted_real_st.mul(None, alus=alu_rq_13, output_streams=g.SG4_E[6])

				# calculate Psi_permuted_imag * gate_off_diag_imag
				state_permuted_imag_st 	   = permuted_state_imag_mt.read( streams=[g.SG4[4]] )
				#gate_off_diag_imag_mt_256 = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_imag_off_diag, name="gate_off_diag_imag_mt_256", initialized=False)					
				#gate_off_diag_imag_st                      = gate_off_diag_imag_mt_256.read( streams=[g.SG4_W[5]] )
				#state_permuted_imag__gate_off_diag_imag_st = g.mul( state_permuted_imag_st, gate_off_diag_imag_st, alus=[5], output_streams=g.SG4_E[5] )
				gate_off_diag_imag_mt.latch(alus=alu_rq_5, input_streams=g.SG4_W[5], time=0)				
				state_permuted_imag__gate_off_diag_imag_st = state_permuted_imag_st.mul(None, alus=alu_rq_5, output_streams=g.SG4_E[5])			

				# calculate Psi_permuted_real * gate_off_diag_real - Psi_permuted_imag * gate_off_diag_imag
				state_permuted__gate_off_diag_real_st	= g.sub( state_permuted_real__gate_off_diag_real_st, state_permuted_imag__gate_off_diag_imag_st, alus=[10], output_streams=g.SG4_E[5]  )	

					
				
				# calculate Psi_permuted_imag * gate_off_diag_real
				#state_permuted_imag__gate_off_diag_real_st	= g.mul( state_permuted_imag_st, gate_off_diag_real_st, alus=[9], output_streams=g.SG4_E[4] )
				gate_off_diag_real_mt.latch(alus=alu_rq_9, input_streams=g.SG4_W[6], time=1)
				state_permuted_imag__gate_off_diag_real_st	= state_permuted_imag_st.mul( None, alus=alu_rq_9, output_streams=g.SG4_E[4] )
					
				# calculate Psi_permuted_real * gate_diag_imag
				#state_permuted_real__gate_off_diag_imag_st	= g.mul( state_permuted_real_st, gate_off_diag_imag_st, alus=[8], output_streams=g.SG4_E[7] )
				gate_off_diag_imag_mt.latch(alus=alu_rq_8, input_streams=g.SG4_W[5], time=1)
				state_permuted_real__gate_off_diag_imag_st	= state_permuted_real_st.mul( None, alus=alu_rq_8, output_streams=g.SG4_E[7] )

				
				# calculate Psi_permuted_real * gate_off_diag_imag + Psi_permuted_imag * gate_off_diag_real
				state_permuted__gate_off_diag_imag_st	= g.add( state_permuted_imag__gate_off_diag_real_st, state_permuted_real__gate_off_diag_imag_st, alus=[11], output_streams=g.SG4_W[7]  )
				
		

				# calculate (Psi*gate_diag).real + (Psi_permuted*gate_off_diag).real
				Psi_transformed_real_st	= g.add( state__gate_diag_real_st,  state_permuted__gate_off_diag_real_st, alus=[7], output_streams=g.SG4_W[2] )
				Psi_transformed_real_st = g.vxm_identity( Psi_transformed_real_st, alus=[6], output_streams=g.SG4_W[2] )

				states_real_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real", storage_req=states_real_mt.storage_request)
				Psi_transformed_real_copy_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real_copy", storage_req=permuted_state_real_mt.storage_request)


				# calculate (Psi*gate_diag).imag + (Psi_permuted*gate_off_diag).imag
				Psi_transformed_imag_st	    = g.add( state__gate_diag_imag_st,  state_permuted__gate_off_diag_imag_st, alus=[14], output_streams=g.SG4_E[6] )
				Psi_transformed_imag_st = g.vxm_identity( Psi_transformed_imag_st, alus=[15], output_streams=g.SG4_W[0] )


				states_imag_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag", storage_req=states_imag_mt.storage_request)
				Psi_transformed_imag_copy_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag_copy", storage_req=permuted_state_imag_mt.storage_request)
				
					
					
			self.pgm_pkg.compile_program_context(pgm5)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
