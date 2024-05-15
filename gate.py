from split_gates import Split_gates
from memory_layouts import *
import groq.api as g

class Gate(Split_gates):
	def __init__(self, pgm_pkg):
		super().__init__(pgm_pkg)
	
	def gate_program(self):
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate"
		
		with self.pgm_pkg.create_program_context(name) as pgm3:

			#g.reserve_tensor(pgm1, pgm2, State_input_real_mt)

			
			# create reusable permutor requests
			permutor_requests = []
			permutor_requests.append( g.tensor.create_permutor_request( [0], 1 ) )
			permutor_requests.append( g.tensor.create_permutor_request( [1], 1 ) )
			permutor_requests.append( g.tensor.create_permutor_request( [0], 1 ) )
			permutor_requests.append( g.tensor.create_permutor_request( [1], 1 ) )

			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			State_input_real_shared = g.shared_memory_tensor(self.State_input_real_mt, name="State_input_real_shared")
			State_input_imag_shared = g.shared_memory_tensor(self.State_input_imag_mt, name="State_input_imag_shared")
			State_input_real_copy_shared = g.shared_memory_tensor(self.State_input_real_copy_mt, name="State_input_real_copy_shared")
			State_input_imag_copy_shared = g.shared_memory_tensor(self.State_input_imag_copy_mt, name="State_input_imag_copy_shared")

			state_permute_map_selector_shared 	= g.shared_memory_tensor(self.state_permute_map_selector_mt, name="state_permute_map_selector_shared")
			
			#state_permute_map_selector_st = state_permute_map_selector_shared.read(streams=g.SG1_W[24])
			
			#state_permute_map_selector_list = g.split_inner_splits(state_permute_map_selector_st)		
			state_permute_map_selector_list = g.split_vectors(input=state_permute_map_selector_shared, splits=[1]*gate_count)
			
			permute_maps_mt_west_shared = g.shared_memory_tensor(self.permute_maps_mt_west, name=f"permute_maps_west_shared")
			permute_maps_mt_east_shared = g.shared_memory_tensor(self.permute_maps_mt_east, name=f"permute_maps_east_shared")

			# state_1_mt: if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
			# state_0_mt: if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			state_1_broadcasted_shared = g.shared_memory_tensor(self.state_1_broadcasted_mt, name="state_1_broadcasted_shared")
			state_0_broadcasted_shared = g.shared_memory_tensor(self.state_0_broadcasted_mt, name="state_0_broadcasted_shared")

			# Creates a shared tensor to reuse the memory allocation made by program pgm2

			# real and imag components of the broadcasted 00, 01, 10, 11 elements of the gate kernels (each element of a 320 vector contains either the 00,01,10 or 11 element)
			gate_kernels_broadcasted_real_shared_dict = {}
			gate_kernels_broadcasted_imag_shared_dict = {}
			gate_kernels_broadcasted_real_shared_copy_dict = {}
			gate_kernels_broadcasted_imag_shared_copy_dict = {}
			for gate_idx in range( 1 ):#gate_count ):
				for label in gate_kernel_element_labels:

					key = f"{gate_idx}_"+label
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

			pre = []
			tm = [0, 118]

			gate_first_column_real_list = []
			gate_first_column_imag_list = []
			gate_second_column_real_list = []
			gate_second_column_imag_list = []
			for gate_idx in range( 1 ):#gate_count ):
			
				address_offset = 4*gate_count + gate_idx

				layout_real_first_column = layout_gate_kernels_real + f", A1({address_offset})"
				layout_imag_first_column = layout_gate_kernels_imag + f", A1({address_offset})"		
				layout_real_second_column = layout_gate_kernels_real_copy + f", A1({address_offset})"
				layout_imag_second_column = layout_gate_kernels_imag_copy + f", A1({address_offset})"	
				
				storage_req_real_first_column  = g.tensor.create_storage_request( layout = layout_real_first_column )
				storage_req_imag_first_column  = g.tensor.create_storage_request( layout = layout_imag_first_column )
				storage_req_real_second_column = g.tensor.create_storage_request( layout = layout_real_second_column )
				storage_req_imag_second_column = g.tensor.create_storage_request( layout = layout_imag_second_column )									

				alu_rq_1  = g.tensor.create_alu_request( alus=[1] )
				alu_rq_12 = g.tensor.create_alu_request( alus=[12] )
							
				with g.ResourceScope(name=f"prepare_state_pair_scope_{gate_idx}", is_buffered=True, time=tm[gate_idx]) as prepare_state_pair_scope :

					# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
					print('gatescope')   
					#target_qbit = 0

					print( g.mem_gather.__doc__ )
					print( permute_maps_mt_west_shared.physical_shape )
					print( permute_maps_mt_east_shared.physical_shape )
					print( state_permute_map_selector_shared.physical_shape )
					#state_permute_map_selector_st = state_permute_map_selector_shared.read(streams=[g.SG1_W[24]])
					print(state_permute_map_selector_list[0].shape)
					
					
					state_permute_map_selector_west_st = state_permute_map_selector_list[gate_idx].read(streams=[g.SG1_W[24]])
					state_permute_map_selector_east_st = state_permute_map_selector_list[gate_idx].read(streams=[g.SG1_E[24]])
					permute_map_west_st                = g.mem_gather(permute_maps_mt_west_shared, state_permute_map_selector_west_st, output_streams=[g.SG1_W[24]])
					permute_map_east_st                = g.mem_gather(permute_maps_mt_east_shared, state_permute_map_selector_east_st, output_streams=[g.SG1_E[24]])

					state_real_mt_8 = g.reinterpret(Psi_transformed_real_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
					state_imag_mt_8 = g.reinterpret(Psi_transformed_imag_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input

					state_real_st_8 = state_real_mt_8.read( streams=g.SG1_E[0] )
					state_imag_st_8 = state_imag_mt_8.read( streams=g.SG1_W[0] )

				
					permuted_state_real_st_8 = g.permute_inner(state_real_st_8, permute_map_west_st, permutor_req=permutor_requests[gate_idx*2+1], input_streams=[g.SG1_E[0], g.SG1_E[24]], output_streams=g.SG1[12], time=0 )
					permuted_state_imag_st_8 = g.permute_inner(state_imag_st_8, permute_map_east_st, permutor_req=permutor_requests[gate_idx*2], input_streams=[g.SG1_W[0], g.SG1_W[24]], output_streams=g.SG1[0], time=0 )
					print(10*"#" + " permuted")

					permuted_state_real_st_8 = g.reshape(permuted_state_real_st_8, [memory_slices, 4, matrix_size] )
					permuted_state_imag_st_8 = g.reshape(permuted_state_imag_st_8, [memory_slices, 4, matrix_size] )
				
					permuted_state_real_st = g.reinterpret(permuted_state_real_st_8, g.float32 )
					permuted_state_imag_st = g.reinterpret(permuted_state_imag_st_8, g.float32 )

					permuted_state_real_mt = permuted_state_real_st.write(name=f"permuted_result_real", layout=layout_state_vector_real_permuted)
					permuted_state_imag_mt = permuted_state_imag_st.write(name=f"permuted_result_imag", layout=layout_state_vector_imag_permuted)

					g.add_mem_constraints(Psi_transformed_real_copy_list, [permuted_state_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					Psi_transformed_real_copy_list.append( permuted_state_real_mt )

					g.add_mem_constraints(Psi_transformed_imag_copy_list, [permuted_state_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					Psi_transformed_imag_copy_list.append( permuted_state_imag_mt )

				
				#################################################################################
				# component to perform the gate operation
				
				with g.ResourceScope(name=f"prepare_gate_kernel_scope_{gate_idx}", is_buffered=True, time=tm[gate_idx]) as prepare_gate_kernel_scope :

		
					# combine 00 and 10 elements of the gate kernel to be used in the transformation
					#states_1_mt_list = g.split( states_1_shared, 1, 0)
					#states_0_mt_list = g.split( states_0_shared, 1, 0)

					#state_1_mt = states_1_mt_list[target_qbit] #TODO: do with gather map
					#state_0_mt = states_0_mt_list[target_qbit] #TODO: do with gather map
					
					
					# filter the elements in the vector according to the indices standig for output states |0> and |1> 
					
					with g.ResourceScope(name=f"prepare_fileterd_gate_elements_scope_{gate_idx}", is_buffered=True,  time=0) as prepare_fileterd_gate_elements_scope :

						state_1_broadcasted_st = state_1_broadcasted_shared.read( streams=g.SG4[1] )
						state_0_broadcasted_st = state_0_broadcasted_shared.read( streams=g.SG4[7] )
				
						# real and imaginary parts of the state concatenated into a unified tensor
						gate_00_real_st = gate_kernels_broadcasted_real_shared_dict[f"{gate_idx}_00"].read( streams=g.SG4[6], time=0 )
						gate_00_imag_st = gate_kernels_broadcasted_imag_shared_dict[f"{gate_idx}_00"].read( streams=g.SG4[6], time=0 )
						gate_00_st 	= g.stack( [gate_00_real_st, gate_00_imag_st], dim=0 )

						#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
						gate_00_st = g.reinterpret( gate_00_st, g.uint32 )
						gate_00_st = g.bitwise_and( gate_00_st, state_0_broadcasted_st, alus=[8], output_streams=g.SG4[4] )
						gate_00_st = g.reinterpret( gate_00_st, g.float32 )




						# real and imaginary parts of the state concatenated into a unified tensor
						gate_10_real_st = gate_kernels_broadcasted_real_shared_copy_dict[f"{gate_idx}_11"].read( streams=g.SG4[0])#, time=2 )
						gate_10_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict[f"{gate_idx}_11"].read( streams=g.SG4[0])#, time=2 )
						gate_10_st 	= g.stack( [gate_10_real_st, gate_10_imag_st], dim=0 )

						gate_10_st = g.reinterpret( gate_10_st, g.uint32 )
						gate_10_st = g.bitwise_and( gate_10_st, state_1_broadcasted_st, alus=[0], output_streams=g.SG4[3] )
						gate_10_st = g.reinterpret( gate_10_st, g.float32 )

				
						# combine the filtered gate element vectors
						gate_first_column_st = g.add(gate_00_st, gate_10_st, alus=[5], output_streams=g.SG4[3])
				
							
						print("layout: ", layout_imag_first_column)
					
						gate_first_column_st_list	= g.split( gate_first_column_st, num_splits=2, dim=0 )
						gate_first_column_real_mt 	= gate_first_column_st_list[0].write(name=f"gate_first_column_real_{gate_idx}", storage_req=storage_req_real_first_column, program_output=True)
						gate_first_column_imag_mt 	= gate_first_column_st_list[1].write(name=f"gate_first_column_imag_{gate_idx}", storage_req=storage_req_imag_first_column, program_output=True)
						#if gate_idx > 0:
						#	g.add_mem_constraints(gate_first_column_real_list, [gate_first_column_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
						#	g.add_mem_constraints(gate_first_column_imag_list, [gate_first_column_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)

						#gate_first_column_real_list.append(gate_first_column_real_mt)
						#gate_first_column_imag_list.append(gate_first_column_imag_mt)

						#gate_for_output_state_0_combined_st = g.add(gate_00_real_st, gate_10_real_st, alus=[5], output_streams=g.SG4[3])
						#state_0_broadcasted_mt = gate_for_output_state_0_combined_st.write(name=f"ttt", layout=f"H1(E), -1, S4", program_output=True)



				#with g.ResourceScope(name="prepare_gate_elements_output_state_1_scope", is_buffered=True, predecessors=[prepare_gate_elements_output_state_0_scope], time=None) as prepare_gate_elements_output_state_1_scope :

						state_1_broadcasted_st = state_1_broadcasted_shared.read( streams=g.SG4[1] )
						state_0_broadcasted_st = state_0_broadcasted_shared.read( streams=g.SG4[7] )
				
						# real and imaginary parts of the state concatenated into a unified tensor
						gate_01_real_st = gate_kernels_broadcasted_real_shared_dict[f"{gate_idx}_01"].read( streams=g.SG4[6])#, time=5 )
						gate_01_imag_st = gate_kernels_broadcasted_imag_shared_dict[f"{gate_idx}_01"].read( streams=g.SG4[6])#, time=5 )
						gate_01_st 	= g.stack( [gate_01_real_st, gate_01_imag_st], dim=0 )

						#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
						gate_01_st = g.reinterpret( gate_01_st, g.uint32 )
						gate_01_st = g.bitwise_and( gate_01_st, state_0_broadcasted_st, alus=[10], output_streams=g.SG4[4] )
						gate_01_st = g.reinterpret( gate_01_st, g.float32 )

						# real and imaginary parts of the state concatenated into a unified tensor
						gate_11_real_st = gate_kernels_broadcasted_real_shared_copy_dict[f"{gate_idx}_10"].read( streams=g.SG4[0], time=7 )
						gate_11_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict[f"{gate_idx}_10"].read( streams=g.SG4[0], time=7 )
						gate_11_st 	= g.stack( [gate_11_real_st, gate_11_imag_st], dim=0 )

						gate_11_st = g.reinterpret( gate_11_st, g.uint32 )
						gate_11_st = g.bitwise_and( gate_11_st, state_1_broadcasted_st, alus=[2], output_streams=g.SG4[3] )
						gate_11_st = g.reinterpret( gate_11_st, g.float32 )
					
						# combine the filtered gate element vectors
						gate_second_column_st = g.add(gate_01_st, gate_11_st, alus=[7], output_streams=g.SG4[3])
				
							
					
						gate_second_column_st_list	= g.split( gate_second_column_st, num_splits=2, dim=0 )
						gate_second_column_real_mt 	= gate_second_column_st_list[0].write(name=f"gate_second_column_real_{gate_idx}", storage_req=storage_req_real_second_column, program_output=True)
						gate_second_column_imag_mt 	= gate_second_column_st_list[1].write(name=f"gate_second_column_imag_{gate_idx}", storage_req=storage_req_imag_second_column, program_output=True)

						if gate_idx > 0:
							g.add_mem_constraints(gate_second_column_real_list, [gate_second_column_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
							g.add_mem_constraints(gate_second_column_imag_list, [gate_second_column_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
						gate_second_column_real_list.append(gate_second_column_real_mt)
						gate_second_column_imag_list.append(gate_second_column_imag_mt)
						#gate_output_state_0_real_copy_mt = gate_output_state_0_st_list[0].write(name=f"gate_output_state_0_real_copy", layout=layout_real_copy, program_output=True)
						#gate_output_state_0_imag_copy_mt = gate_output_state_0_st_list[1].write(name=f"gate_output_state_0_imag_copy", layout=layout_imag_copy, program_output=True)
						#gate_output_state_0_copy_mt = gate_output_state_0_st.write(name=f"gate_output_state_copy_0", layout=layout_real_copy)			
					

						#gate_for_output_state_0_combined_st = g.add(gate_00_real_st, gate_10_real_st, alus=[5], output_streams=g.SG4[3])
						#state_0_broadcasted_mt = gate_for_output_state_0_combined_st.write(name=f"ttt", layout=f"H1(E), -1, S4", program_output=True)
				
				#################################################################################
				
				with g.ResourceScope(name=f"apply_gate_scope_{gate_idx}", is_buffered=True, predecessors=[prepare_state_pair_scope], time=None) as apply_gate_scope :


					# calculate Psi_real * gate_first_column_real				
					state_real_st 			      = Psi_transformed_real_mt.read( streams=[g.SG4[0]] )	
					gate_first_column_real_mt_256         = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_real_first_column, name="gate_first_column_real_mt_256", initialized=False)	
					gate_first_column_real_st 		      = gate_first_column_real_mt_256.read( streams=[g.SG4_W[0]] )
					state_real__gate_first_column_real_st = g.mul( state_real_st, gate_first_column_real_st, alus=alu_rq_12, output_streams=g.SG4_E[0], time=150 )
					


					# calculate Psi_imag * gate_first_column_imag
					state_imag_st 			                = Psi_transformed_imag_mt.read( streams=[g.SG4[3]] )
					gate_first_column_imag_mt_256         = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_imag_first_column, name="gate_first_column_imag_mt_256", initialized=False)	
					gate_first_column_imag_st 		        = gate_first_column_imag_mt_256.read( streams=[g.SG4_W[3]] )
					state_imag__gate_first_column_imag_st	= g.mul( state_imag_st, gate_first_column_imag_st, alus=[4], output_streams=g.SG4_E[2] )			

					# calculate Psi_real * gate_first_column_real - Psi_imag * gate_first_column_imag
					state__gate_first_column_real_st	= g.sub( state_real__gate_first_column_real_st, state_imag__gate_first_column_imag_st, alus=[2], output_streams=g.SG4_E[2] )
					
					
					#res_mt = state__gate_first_column_real_st.write(name=f"test", layout="H1(E), -1, S4", program_output=True)

					
					
					
					# calculate Psi_imag * gate_first_column_real
					state_imag__gate_first_column_real_st	= g.mul( state_imag_st, gate_first_column_real_st, alus=alu_rq_1, output_streams=g.SG4_E[3] )			
					
					# calculate Psi_real * gate_first_column_imag
					state_real__gate_first_column_imag_st	= g.mul( state_real_st, gate_first_column_imag_st, alus=[0], output_streams=g.SG4_E[1] )

					# calculate Psi_real * gate_first_column_imag + Psi_imag * gate_first_column_real
					state__gate_first_column_imag_st	= g.add( state_imag__gate_first_column_real_st, state_real__gate_first_column_imag_st, alus=[3], output_streams=g.SG4_W[1]  )
				
					#res3_mt = state__gate_first_column_imag_st.write(name=f"test3", layout="H1, -1, S4", program_output=True)
					
		# alus 8, 13			
					
					
					# calculate Psi_permuted_real * gate_second_column_real
					state_permuted_real_st 			            	= permuted_state_real_mt.read( streams=[g.SG4_E[6]] )
					gate_second_column_real_mt_256                  = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_real_second_column, name="gate_second_column_real_mt_256", initialized=False)					
					gate_second_column_real_st 			         	= gate_second_column_real_mt_256.read( streams=[g.SG4_W[6]] )
					state_permuted_real__gate_second_column_real_st	= g.mul( state_permuted_real_st, gate_second_column_real_st, alus=[13], output_streams=g.SG4_E[6] )

					# calculate Psi_permuted_imag * gate_second_column_imag
					state_permuted_imag_st 				            = permuted_state_imag_mt.read( streams=[g.SG4[4]] )
					gate_second_column_imag_mt_256                  = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_imag_second_column, name="gate_second_column_imag_mt_256", initialized=False)					
					gate_second_column_imag_st                      = gate_second_column_imag_mt_256.read( streams=[g.SG4[5]] )
					state_permuted_imag__gate_second_column_imag_st	= g.mul( state_permuted_imag_st, gate_second_column_imag_st, alus=[5], output_streams=g.SG4_E[5] )			

					# calculate Psi_permuted_real * gate_second_column_real - Psi_permuted_imag * gate_second_column_imag
					state_permuted__gate_second_column_real_st	= g.sub( state_permuted_real__gate_second_column_real_st, state_permuted_imag__gate_second_column_imag_st, alus=[10], output_streams=g.SG4_E[5]  )
					#res_mt = state_permuted__gate_second_column_real_st.write(name=f"test2", layout="H1(E), -1, S4", program_output=True)

					
				
					# calculate Psi_permuted_imag * gate_second_column_real
					state_permuted_imag__gate_second_column_real_st	= g.mul( state_permuted_imag_st, gate_second_column_real_st, alus=[9], output_streams=g.SG4_E[4] )
					#res_mt = state_permuted_imag__gate_second_column_real_st.write(name=f"test4", layout="H1(E), -1, S4(20-28)", program_output=True)
					
					# calculate Psi_permuted_real * gate_first_column_imag
					state_permuted_real__gate_second_column_imag_st	= g.mul( state_permuted_real_st, gate_second_column_imag_st, alus=[8], output_streams=g.SG4_E[7] )
					#res_mt = state_permuted_real__gate_second_column_imag_st.write(name=f"test5", layout="H1(E), -1, S4(10-20)", program_output=True)

				
					# calculate Psi_permuted_real * gate_second_column_imag + Psi_permuted_imag * gate_second_column_real
					state_permuted__gate_second_column_imag_st	= g.add( state_permuted_imag__gate_second_column_real_st, state_permuted_real__gate_second_column_imag_st, alus=[11], output_streams=g.SG4_W[7]  )
				
					#res4_mt = state_permuted__gate_second_column_imag_st.write(name=f"test2", layout="H1(W), -1, S4", program_output=True)	
					

					# number of memory slices to store the transformed state
					#slice_offset  = ((gate_idx+1) % 2) * 4096
		

					layout_transformed_state_real = layout_State_input_real#f"H1(W), A{memory_slices}, S4(12-15)"#
					layout_transformed_state_imag = layout_State_input_imag#f"H1(W), A{memory_slices}, S4(8-11)"#layout_State_input_imag + f", A{memory_slices}({slice_offset}-{slice_offset+memory_slices-1})"

					layout_transformed_state_real_copy = layout_State_input_real_copy
					layout_transformed_state_imag_copy = layout_State_input_imag_copy

					print( g.vxm_identity.__doc__ )

					# calculate (Psi*gate_first_column).real + (Psi_permuted*gate_second_column).real
					Psi_transformed_real_st	= g.add( state__gate_first_column_real_st,  state_permuted__gate_second_column_real_st, alus=[7], output_streams=g.SG4_W[2] )
					Psi_transformed_real_st = g.vxm_identity( Psi_transformed_real_st, alus=[6], output_streams=g.SG4_W[2] )

					Psi_transformed_real_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real_{gate_idx}", layout=layout_transformed_state_real, program_output=True)
					#Psi_transformed_real_copy_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real_copy_{gate_idx}", layout=layout_transformed_state_real_copy)

					g.add_mem_constraints(Psi_transformed_real_list, [Psi_transformed_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					Psi_transformed_real_list.append( Psi_transformed_real_mt )
				
					#g.add_mem_constraints(Psi_transformed_real_copy_list, [Psi_transformed_real_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					#Psi_transformed_real_copy_list.append( Psi_transformed_real_copy_mt )

					# calculate (Psi*gate_first_column).imag + (Psi_permuted*gate_second_column).imag
					Psi_transformed_imag_st	    = g.add( state__gate_first_column_imag_st,  state_permuted__gate_second_column_imag_st, alus=[14], output_streams=g.SG4_E[6] )
					Psi_transformed_imag_st = g.vxm_identity( Psi_transformed_imag_st, alus=[15], output_streams=g.SG4_W[0] )



					Psi_transformed_imag_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag_{gate_idx}", layout=layout_transformed_state_imag, program_output=True)
					#Psi_transformed_imag_copy_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag_copy_{gate_idx}", layout=layout_transformed_state_imag_copy)
					
					g.add_mem_constraints(Psi_transformed_imag_list, [Psi_transformed_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					Psi_transformed_imag_list.append( Psi_transformed_imag_mt )
					
					#g.add_mem_constraints(Psi_transformed_imag_copy_list, [Psi_transformed_imag_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					#Psi_transformed_imag_copy_list.append( Psi_transformed_imag_copy_mt )
					
					
			self.pgm_pkg.compile_program_context(pgm3)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
