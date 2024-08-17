from split_gates import Split_gates
from memory_layouts import *
import groq.api as g

class Gate(Split_gates):
	def __init__(self, pgm_pkg, mxm_permutation=True):
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation)
	
	def gate_program(self):
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate"
		
		with self.pgm_pkg.create_program_context(name) as pgm3:

			#g.reserve_tensor(pgm1, pgm2, State_input_real_mt)
				
			control_qbit = [9]

			# create reusable permutor requests
			permutor_requests = []
			permutor_requests.append( g.tensor.create_permutor_request( [0], 1 ) )
			permutor_requests.append( g.tensor.create_permutor_request( [1], 1 ) )
			permutor_requests.append( g.tensor.create_permutor_request( [0], 1 ) )
			permutor_requests.append( g.tensor.create_permutor_request( [1], 1 ) )

			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			State_input_real_shared      = g.shared_memory_tensor(self.State_input_real_mt, name="State_input_real_shared")
			State_input_imag_shared      = g.shared_memory_tensor(self.State_input_imag_mt, name="State_input_imag_shared")
			State_input_real_copy_shared = g.shared_memory_tensor(self.State_input_real_copy_mt, name="State_input_real_copy_shared")
			State_input_imag_copy_shared = g.shared_memory_tensor(self.State_input_imag_copy_mt, name="State_input_imag_copy_shared")

			state_permute_map_selector_shared = g.shared_memory_tensor(self.state_permute_map_selector_mt, name="state_permute_map_selector_shared")
			
			state_permute_map_selector_list   = g.split_vectors(input=state_permute_map_selector_shared, splits=[1]*gate_count)
			
			permute_maps_mt_west_shared = g.shared_memory_tensor(self.permute_maps_mt_west, name=f"permute_maps_west_shared")
			permute_maps_mt_east_shared = g.shared_memory_tensor(self.permute_maps_mt_east, name=f"permute_maps_east_shared")

			# state_1_mt: if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
			# state_0_mt: if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			state_1_broadcasted_shared = g.shared_memory_tensor(self.state_1_broadcasted_mt, name="state_1_broadcasted_shared")
			state_0_broadcasted_shared = g.shared_memory_tensor(self.state_0_broadcasted_mt, name="state_0_broadcasted_shared")

			ones_shared   = g.shared_memory_tensor(self.ones_mt, name="ones_shared")
			zeros_shared  = g.shared_memory_tensor(self.zeros_mt, name="zeros_shared")
			ones_zeros_mt = g.concat([ones_shared, zeros_shared], dim=0, name="ones_zeros_mt")


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



			# the transformed states are interated further from gate to gate, but for the first gate they are set to the initial state
			Psi_transformed_real_mt = State_input_real_shared
			Psi_transformed_imag_mt = State_input_imag_shared
			Psi_transformed_real_copy_mt = State_input_real_copy_shared
			Psi_transformed_imag_copy_mt = State_input_imag_copy_shared

			# list of tensors for mutual memory exclusions
			Psi_transformed_real_list = [Psi_transformed_real_mt]
			Psi_transformed_imag_list = [Psi_transformed_imag_mt]
			Psi_transformed_real_copy_list = [Psi_transformed_real_copy_mt]
			Psi_transformed_imag_copy_list = [Psi_transformed_imag_copy_mt]


			gate_diag_real_list = []
			gate_diag_imag_list = []
			gate_off_diag_real_list = []
			gate_off_diag_imag_list = []

			Psi_transformed_real_mt_list = g.split( Psi_transformed_real_mt, 1 )
			Psi_transformed_imag_mt_list = g.split( Psi_transformed_imag_mt, 1 )

			Psi_transformed_real_copy_mt_list = g.split( Psi_transformed_real_copy_mt, 1 )
			Psi_transformed_imag_copy_mt_list = g.split( Psi_transformed_imag_copy_mt, 1 )

			index_pair = 1 << (control_qbit[0] - small_qbit_num_limit)

			half_memory_slices = memory_slices >> 1

			stagger_time = 20

			if (stagger_time % 2 ) > 0:
				raise('Stegger time should be a multiple of 2')

			stagger_cycles = half_memory_slices // stagger_time
			if ( memory_slices % stagger_time ) > 0:
				stagger_cycles = stagger_cycles + 1 

			mask_lower_bits = index_pair - 1 
			mask_upper_bits = 1 << ( qbit_num - control_qbit[0] )
			mask_upper_bits = mask_upper_bits - 1
			mask_upper_bits = mask_upper_bits << (control_qbit[0] - small_qbit_num_limit )

			
			address_offset = 4*gate_count

			layout_real_diag = layout_gate_kernels_real + f", A1({address_offset})"
			layout_imag_diag = layout_gate_kernels_imag + f", A1({address_offset})"		
			layout_real_off_diag = layout_gate_kernels_real_copy + f", A1({address_offset})"
			layout_imag_off_diag = layout_gate_kernels_imag_copy + f", A1({address_offset})"	
				
			storage_req_real_diag  = g.tensor.create_storage_request( layout = layout_real_diag )
			storage_req_imag_diag  = g.tensor.create_storage_request( layout = layout_imag_diag )
			storage_req_real_off_diag = g.tensor.create_storage_request( layout = layout_real_off_diag )
			storage_req_imag_off_diag = g.tensor.create_storage_request( layout = layout_imag_off_diag )	

			alu_rq_1  = g.tensor.create_alu_request( alus=[1] )
			alu_rq_12 = g.tensor.create_alu_request( alus=[12] )
							
			with g.ResourceScope(name=f"prepare_state_pair_scope", is_buffered=True, time=0) as prepare_state_pair_scope :

				# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
				print('gatescope')   
					
				state_permute_map_selector_east_st = state_permute_map_selector_list[0].read(streams=[g.SG1_E[24]])
				state_permute_map_selector_west_st = state_permute_map_selector_list[0].read(streams=[g.SG1_W[24]])

				permute_map_west_st = g.mem_gather(permute_maps_mt_west_shared, state_permute_map_selector_west_st, output_streams=[g.SG1_W[24]])
				permute_map_east_st = g.mem_gather(permute_maps_mt_east_shared, state_permute_map_selector_east_st, output_streams=[g.SG1_E[24]])

				states_real_mt_list = []
				states_imag_mt_list = []
				states_real_copy_mt_list = []
				states_imag_copy_mt_list = []
				index = 0
				for outer_idx in range(stagger_cycles):

					index_loc = index
					# gathering the elements where the control qbit is 1
					for stagger_idx in range(stagger_time):

						lower_bits  = index_loc & mask_lower_bits
						higher_bits = index_loc & mask_upper_bits

						state_index_control_qubit_0 = (higher_bits << 1) + lower_bits
						state_index_control_qubit_1 = state_index_control_qubit_0 + index_pair

						states_real_mt_list.append(Psi_transformed_real_mt_list[state_index_control_qubit_1])
						states_imag_mt_list.append(Psi_transformed_imag_mt_list[state_index_control_qubit_1])
						states_real_copy_mt_list.append(Psi_transformed_real_copy_mt_list[state_index_control_qubit_1])
						states_imag_copy_mt_list.append(Psi_transformed_imag_copy_mt_list[state_index_control_qubit_1])

						index_loc += 1

						if index_loc >= half_memory_slices:
							break
					
					index = index_loc

					if index >= half_memory_slices:
						break

				states_real_mt      = g.concat(states_real_mt_list)
				states_imag_mt      = g.concat(states_imag_mt_list)
				states_real_copy_mt = g.concat(states_real_copy_mt_list)
				states_imag_copy_mt = g.concat(states_imag_copy_mt_list)

				storage_req_real = states_real_copy_mt.storage_request
				storage_req_imag = states_imag_copy_mt.storage_request

				# reinterpreting the states for the permutation
				states_real_mt_8 = g.reinterpret(states_real_mt, g.uint8)
				states_imag_mt_8 = g.reinterpret(states_imag_mt, g.uint8)

				states_real_st_8 = states_real_mt_8.read( streams=g.SG1_E[0])
				states_imag_st_8 = states_imag_mt_8.read( streams=g.SG1_W[0])

				permuted_state_real_st_8 = g.permute_inner(states_real_st_8, permute_map_west_st, permutor_req=permutor_requests[1], input_streams=[g.SG1_E[0], g.SG1_E[24]], output_streams=g.SG1[12], time=0 )
				permuted_state_imag_st_8 = g.permute_inner(states_imag_st_8, permute_map_east_st, permutor_req=permutor_requests[0], input_streams=[g.SG1_W[0], g.SG1_W[24]], output_streams=g.SG1[0], time=0)
				print(10*"#" + " permuted")

				# We only work with the half of the state vecor
				permuted_state_real_st_8 = g.reshape(permuted_state_real_st_8, [(memory_slices >> 1), 4, matrix_size] )
				permuted_state_imag_st_8 = g.reshape(permuted_state_imag_st_8, [(memory_slices >> 1), 4, matrix_size] )
				
				permuted_state_real_st = g.reinterpret(permuted_state_real_st_8, g.float32 )
				permuted_state_imag_st = g.reinterpret(permuted_state_imag_st_8, g.float32 )

				permuted_state_real_mt = permuted_state_real_st.write(name=f"permuted_result_real_half", storage_req=storage_req_real)
				permuted_state_imag_mt = permuted_state_imag_st.write(name=f"permuted_result_imag_half", storage_req=storage_req_imag)
					

				
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


							
					print("layout: ", layout_imag_diag)
					
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
					gate_off_diag_real_mt 	= gate_off_diag_st_list[0].write(name=f"gate_off_diag_real", storage_req=storage_req_real_off_diag, program_output=True)
					gate_off_diag_imag_mt 	= gate_off_diag_st_list[1].write(name=f"gate_off_diag_imag", storage_req=storage_req_imag_off_diag, program_output=True)

					gate_off_diag_real_list.append(gate_off_diag_real_mt)
					gate_off_diag_imag_list.append(gate_off_diag_imag_mt)

			#################################################################################
				
			with g.ResourceScope(name=f"apply_gate_scope", is_buffered=True, predecessors=[prepare_state_pair_scope], time=None) as apply_gate_scope :


				# calculate Psi_real * gate_diag_real				
				state_real_st         = states_real_mt.read( streams=[g.SG4[0]] )	
				gate_diag_real_mt_256 = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_real_diag, name="gate_diag_real_mt_256", initialized=False)	
				gate_diag_real_st     = gate_diag_real_mt_256.read( streams=[g.SG4_W[0]] )
				state_real__gate_diag_real_st = g.mul( state_real_st, gate_diag_real_st, alus=alu_rq_12, output_streams=g.SG4_E[0], time=150 )
					


				# calculate Psi_imag * gate_diag_imag
				state_imag_st 		      = states_imag_mt.read( streams=[g.SG4[3]] )
				gate_diag_imag_mt_256         = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_imag_diag, name="gate_diag_imag_mt_256", initialized=False)	
				gate_diag_imag_st 	      = gate_diag_imag_mt_256.read( streams=[g.SG4_W[3]] )
				state_imag__gate_diag_imag_st = g.mul( state_imag_st, gate_diag_imag_st, alus=[4], output_streams=g.SG4_E[2] )			

				# calculate Psi_real * gate_diag_real - Psi_imag * gate_diag_imag
				state__gate_diag_real_st      = g.sub( state_real__gate_diag_real_st, state_imag__gate_diag_imag_st, alus=[2], output_streams=g.SG4_E[2] )
					
					
				#res_mt = state__gate_diag_real_st.write(name=f"test", layout="H1(E), -1, S4", program_output=True)

					
					
					
				# calculate Psi_imag * gate_diag_real
				state_imag__gate_diag_real_st = g.mul( state_imag_st, gate_diag_real_st, alus=alu_rq_1, output_streams=g.SG4_E[3] )			
					
				# calculate Psi_real * gate_diag_imag
				state_real__gate_diag_imag_st = g.mul( state_real_st, gate_diag_imag_st, alus=[0], output_streams=g.SG4_E[1] )

				# calculate Psi_real * gate_diag_imag + Psi_imag * gate_diag_real
				state__gate_diag_imag_st      = g.add( state_imag__gate_diag_real_st, state_real__gate_diag_imag_st, alus=[3], output_streams=g.SG4_W[1]  )
				
				#res3_mt = state__gate_diag_imag_st.write(name=f"test3", layout="H1, -1, S4", program_output=True)
					
		# alus 8, 13			
					
					
				# calculate Psi_permuted_real * gate_off_diag_real
				state_permuted_real_st 	  = permuted_state_real_mt.read( streams=[g.SG4_E[6]] )
				gate_off_diag_real_mt_256 = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_real_off_diag, name="gate_off_diag_real_mt_256", initialized=False)					
				gate_off_diag_real_st 	  = gate_off_diag_real_mt_256.read( streams=[g.SG4_W[6]] )
				state_permuted_real__gate_off_diag_real_st = g.mul( state_permuted_real_st, gate_off_diag_real_st, alus=[13], output_streams=g.SG4_E[6] )

				# calculate Psi_permuted_imag * gate_off_diag_imag
				state_permuted_imag_st 	  = permuted_state_imag_mt.read( streams=[g.SG4[4]] )
				gate_off_diag_imag_mt_256 = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_imag_off_diag, name="gate_off_diag_imag_mt_256", initialized=False)					
				gate_off_diag_imag_st     = gate_off_diag_imag_mt_256.read( streams=[g.SG4_W[5]] )
				state_permuted_imag__gate_off_diag_imag_st = g.mul( state_permuted_imag_st, gate_off_diag_imag_st, alus=[5], output_streams=g.SG4_E[5] )			

				# calculate Psi_permuted_real * gate_off_diag_real - Psi_permuted_imag * gate_off_diag_imag
				state_permuted__gate_off_diag_real_st = g.sub( state_permuted_real__gate_off_diag_real_st, state_permuted_imag__gate_off_diag_imag_st, alus=[10], output_streams=g.SG4_E[5]  )					
				
				# calculate Psi_permuted_imag * gate_off_diag_real
				state_permuted_imag__gate_off_diag_real_st = g.mul( state_permuted_imag_st, gate_off_diag_real_st, alus=[9], output_streams=g.SG4_E[4] )
					
				# calculate Psi_permuted_real * gate_diag_imag
				state_permuted_real__gate_off_diag_imag_st = g.mul( state_permuted_real_st, gate_off_diag_imag_st, alus=[8], output_streams=g.SG4_E[7] )

				
				# calculate Psi_permuted_real * gate_off_diag_imag + Psi_permuted_imag * gate_off_diag_real
				state_permuted__gate_off_diag_imag_st = g.add( state_permuted_imag__gate_off_diag_real_st, state_permuted_real__gate_off_diag_imag_st, alus=[11], output_streams=g.SG4_W[7]  )
					
					

				# number of memory slices to store the transformed state		

				layout_transformed_state_real_copy = layout_State_input_real_copy
				layout_transformed_state_imag_copy = layout_State_input_imag_copy


				# calculate (Psi*gate_diag).real + (Psi_permuted*gate_off_diag).real
				Psi_transformed_real_st	= g.add( state__gate_diag_real_st,  state_permuted__gate_off_diag_real_st, alus=[7], output_streams=g.SG4_W[2] )
				Psi_transformed_real_st = g.vxm_identity( Psi_transformed_real_st, alus=[6], output_streams=g.SG4_W[2] )

				Psi_transformed_real_st.write(name=f"Psi_transformed_real_half", storage_req=states_real_mt.storage_request)
				Psi_transformed_real_st.write(name=f"Psi_transformed_real_copy_half", storage_req=permuted_state_real_mt.storage_request)
				Psi_transformed_real_mt = g.constant_tensor(Psi_transformed_real_mt.shape, g.float32, storage_req=self.state_real_request, name=f"Psi_transformed_real", initialized=False)
				Psi_transformed_real_copy_mt = g.constant_tensor(Psi_transformed_real_copy_mt.shape, g.float32, storage_req=self.state_real_copy_request, name=f"Psi_transformed_real_copy", initialized=False)

				g.add_mem_constraints(Psi_transformed_real_list, [Psi_transformed_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
				Psi_transformed_real_list.append( Psi_transformed_real_mt )
				
				g.add_mem_constraints(Psi_transformed_real_copy_list, [Psi_transformed_real_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
				Psi_transformed_real_copy_list.append( Psi_transformed_real_copy_mt )

				# calculate (Psi*gate_diag).imag + (Psi_permuted*gate_off_diag).imag
				Psi_transformed_imag_st	    = g.add( state__gate_diag_imag_st,  state_permuted__gate_off_diag_imag_st, alus=[14], output_streams=g.SG4_E[6] )
				Psi_transformed_imag_st = g.vxm_identity( Psi_transformed_imag_st, alus=[15], output_streams=g.SG4_W[0] )


				Psi_transformed_imag_st.write(name=f"Psi_transformed_imag_half", storage_req=states_imag_mt.storage_request)
				Psi_transformed_imag_st.write(name=f"Psi_transformed_imag_copy_half", storage_req=permuted_state_imag_mt.storage_request)
				Psi_transformed_imag_mt = g.constant_tensor(Psi_transformed_imag_mt.shape, g.float32, storage_req=self.state_imag_request, name=f"Psi_transformed_imag", initialized=False)
				Psi_transformed_imag_copy_mt = g.constant_tensor(Psi_transformed_imag_copy_mt.shape, g.float32, storage_req=self.state_imag_copy_request, name=f"Psi_transformed_imag_copy", initialized=False)

				Psi_transformed_real_mt.set_program_output()
				Psi_transformed_imag_mt.set_program_output()
					
				g.add_mem_constraints(Psi_transformed_imag_list, [Psi_transformed_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
				Psi_transformed_imag_list.append( Psi_transformed_imag_mt )
					
				g.add_mem_constraints(Psi_transformed_imag_copy_list, [Psi_transformed_imag_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
				Psi_transformed_imag_copy_list.append( Psi_transformed_imag_copy_mt )
					
					
			self.pgm_pkg.compile_program_context(pgm3)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
