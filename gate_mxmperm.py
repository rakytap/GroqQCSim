from split_gates import Split_gates
from memory_layouts import *
import groq.api as g
import numpy as np

class Gate(Split_gates):
	def __init__(self, pgm_pkg):
		super().__init__(pgm_pkg)
	
	def gate_program(self):
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_mxmperm"
		
		with self.pgm_pkg.create_program_context(name) as pgm3:

			#g.reserve_tensor(pgm1, pgm2, State_input_real_mt)

			identity = np.identity(320, dtype=np.int8)
			# TODO: make layout, try to do it no one slice, move this to upload.py
			identity_west_mt = g.from_data(identity, layout="H1(W), -1, S16")
			identity_east_mt = g.from_data(identity, layout="H1(E), -1, S16")
			
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
			tm_pgk = [50, 168]

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

				mxm_requests = []
				mxm_requests.append(g.tensor.create_mxm_request(planes=[0], signed=True))
				mxm_requests.append(g.tensor.create_mxm_request(planes=[1], signed=True))
				mxm_requests.append(g.tensor.create_mxm_request(planes=[2], signed=True))
				mxm_requests.append(g.tensor.create_mxm_request(planes=[3], signed=True))
							
				with g.ResourceScope(name=f"prepare_state_pair_scope_{gate_idx}", is_buffered=True, time=tm[gate_idx]) as prepare_state_pair_scope :

					# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
					print('gatescope')   
					target_qbit = 2 # TODO: make it not hardcoded

					target_qbit_pair_diff = 1 << target_qbit
					print('target_qubit_pair_diff', target_qbit_pair_diff)
					matrix_row_indexes = []
					for idx in range(320):
						if (idx < matrix_size):
							matrix_row_indexes.append(idx ^ target_qbit_pair_diff)
						else:
							matrix_row_indexes.append(idx)
					matrix_row_indexes = np.array(matrix_row_indexes, dtype=np.int32)

					identity_west_mt_list = np.array(g.split(identity_west_mt, 1))
					identity_east_mt_list = np.array(g.split(identity_east_mt, 1))
					identity_west_mt_list = identity_west_mt_list[matrix_row_indexes]
					identity_east_mt_list = identity_east_mt_list[matrix_row_indexes]

					permute_matrix_west_mt = g.concat(identity_west_mt_list)
					permute_matrix_east_mt = g.concat(identity_east_mt_list)


					# mxm only accepts tensors with an inner dim of 320
					Psi_transformed_real_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=Psi_transformed_real_mt.storage_request, initialized=False)
					Psi_transformed_imag_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=Psi_transformed_imag_mt.storage_request, initialized=False)

					Psi_transformed_real_mt_320 = g.reinterpret(Psi_transformed_real_mt_320, dtype=g.int8)
					Psi_transformed_imag_mt_320 = g.reinterpret(Psi_transformed_imag_mt_320, dtype=g.int8)

					Psi_transformed_real_copy_mt = g.reinterpret(Psi_transformed_real_copy_mt, dtype=g.int8)
					Psi_transformed_imag_copy_mt = g.reinterpret(Psi_transformed_imag_copy_mt, dtype=g.int8)

					# separating the tensors along the slice dimension
					Psi_transformed_real_mt_list = g.split(Psi_transformed_real_mt_320, dim=-2, num_splits=g.float32.size)
					Psi_transformed_imag_mt_list = g.split(Psi_transformed_imag_mt_320, dim=-2, num_splits=g.float32.size)
					Psi_transformed_real_copy_mt_list = g.split(Psi_transformed_real_copy_mt, dim=-2, num_splits=g.float32.size)
					Psi_transformed_imag_copy_mt_list = g.split(Psi_transformed_imag_copy_mt, dim=-2, num_splits=g.float32.size)
					

					time_list = [8, 28, 8+memory_slices, 28+memory_slices]
					weights_list = []
					weights_list.append(g.install_weights(permute_matrix_west_mt, planes=mxm_requests[0], time=0))
					weights_list.append(g.install_weights(permute_matrix_west_mt, planes=mxm_requests[1], time=20))
					weights_list.append(g.install_weights(permute_matrix_east_mt, planes=mxm_requests[2], time=0))
					weights_list.append(g.install_weights(permute_matrix_east_mt, planes=mxm_requests[3], time=20))
					for byte_idx in range(4):
						# Real part
						permuted_state_real_st = Psi_transformed_real_mt_list[byte_idx].matmul(weights_list[byte_idx%2+2], planes=mxm_requests[byte_idx%2+2], time=time_list[byte_idx])
						permuted_state_real_st = g.reinterpret(permuted_state_real_st, dtype=g.int8)
						permuted_state_real_st = g.split(permuted_state_real_st, dim=-2, num_splits=g.int32.size)[0]
						permuted_state_real_st.write(storage_req=Psi_transformed_real_copy_mt_list[byte_idx].storage_request)

						# Imag part
						permuted_state_imag_st = Psi_transformed_imag_mt_list[byte_idx].matmul(weights_list[byte_idx%2], planes=mxm_requests[byte_idx%2], time=time_list[byte_idx])
						permuted_state_imag_st = g.reinterpret(permuted_state_imag_st, dtype=g.int8)
						permuted_state_imag_st = g.split(permuted_state_imag_st, dim=-2, num_splits=g.int32.size)[0]
						permuted_state_imag_st.write(storage_req=Psi_transformed_imag_copy_mt_list[byte_idx].storage_request)

					Psi_transformed_real_copy_mt = g.reinterpret(Psi_transformed_real_copy_mt, dtype=g.float32)
					Psi_transformed_imag_copy_mt = g.reinterpret(Psi_transformed_imag_copy_mt, dtype=g.float32)

					permuted_state_real_mt = g.constant_tensor(Psi_transformed_real_copy_mt.shape, g.float32, name="permuted_result_real", storage_req=self.state_real_copy_request, initialized=False)
					permuted_state_imag_mt = g.constant_tensor(Psi_transformed_imag_copy_mt.shape, g.float32, name="permuted_result_imag", storage_req=self.state_imag_copy_request, initialized=False)
					
					print(10*"#" + " permuted")

					# permuted_state_real_st_8 = g.reshape(permuted_state_real_st_8, [memory_slices, 4, matrix_size] )
					# permuted_state_imag_st_8 = g.reshape(permuted_state_imag_st_8, [memory_slices, 4, matrix_size] )
					# 			
					# permuted_state_real_st = g.reinterpret(permuted_state_real_st_8, g.float32 )
					# permuted_state_imag_st = g.reinterpret(permuted_state_imag_st_8, g.float32 )
					#
					# permuted_state_real_mt = permuted_state_real_st.write(name=f"permuted_result_real", storage_req=self.state_real_copy_request)
					# permuted_state_imag_mt = permuted_state_imag_st.write(name=f"permuted_result_imag", storage_req=self.state_imag_copy_request)

					g.add_mem_constraints(Psi_transformed_real_copy_list, [permuted_state_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					Psi_transformed_real_copy_list.append( permuted_state_real_mt )
					
					g.add_mem_constraints(Psi_transformed_imag_copy_list, [permuted_state_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					Psi_transformed_imag_copy_list.append( permuted_state_imag_mt )





				
				#################################################################################
				# component to perform the gate operation
				# because loading the weights uses 16 streams had to move this after that
				# TODO: solve the stream conflicts to enable doing these parallel 
				
				with g.ResourceScope(name=f"prepare_gate_kernel_scope_{gate_idx}", is_buffered=True, time=tm_pgk[gate_idx]) as prepare_gate_kernel_scope :

		
					# combine 00 and 10 elements of the gate kernel to be used in the transformation
					#states_1_mt_list = g.split( states_1_shared, 1, 0)
					#states_0_mt_list = g.split( states_0_shared, 1, 0)

					#state_1_mt = states_1_mt_list[target_qbit] #TODO: do with gather map
					#state_0_mt = states_0_mt_list[target_qbit] #TODO: do with gather map
					
					
					# filter the elements in the vector according to the indices standig for output states |0> and |1> 
					
					with g.ResourceScope(name=f"prepare_fileterd_gate_elements_scope_{gate_idx}", is_buffered=True,  time=0) as prepare_fileterd_gate_elements_scope :

						state_1_broadcasted_st = state_1_broadcasted_shared.read( streams=g.SG4[2] )
						state_0_broadcasted_st = state_0_broadcasted_shared.read( streams=g.SG4[0] )
				
						# real and imaginary parts of the state concatenated into a unified tensor
						gate_00_real_st = gate_kernels_broadcasted_real_shared_dict[f"{gate_idx}_00"].read( streams=g.SG4[1], time=0 )
						gate_00_imag_st = gate_kernels_broadcasted_imag_shared_dict[f"{gate_idx}_00"].read( streams=g.SG4[1], time=0 )
						gate_00_st 	= g.stack( [gate_00_real_st, gate_00_imag_st], dim=0 )

						#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
						gate_00_st = g.reinterpret( gate_00_st, g.uint32 )
						gate_00_st = g.bitwise_and( gate_00_st, state_0_broadcasted_st, alus=[12], output_streams=g.SG4[0] )
						gate_00_st = g.reinterpret( gate_00_st, g.float32 )




						# real and imaginary parts of the state concatenated into a unified tensor
						gate_10_real_st = gate_kernels_broadcasted_real_shared_copy_dict[f"{gate_idx}_11"].read( streams=g.SG4[3])#, time=2 )
						gate_10_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict[f"{gate_idx}_11"].read( streams=g.SG4[3])#, time=2 )
						gate_10_st 	= g.stack( [gate_10_real_st, gate_10_imag_st], dim=0 )

						gate_10_st = g.reinterpret( gate_10_st, g.uint32 )
						gate_10_st = g.bitwise_and( gate_10_st, state_1_broadcasted_st, alus=[4], output_streams=g.SG4[2] )
						gate_10_st = g.reinterpret( gate_10_st, g.float32 )

				
						# combine the filtered gate element vectors
						gate_first_column_st = g.add(gate_00_st, gate_10_st, alus=[1], output_streams=g.SG4[3])
				
							
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

						state_1_broadcasted_st = state_1_broadcasted_shared.read( streams=g.SG4[2] )
						state_0_broadcasted_st = state_0_broadcasted_shared.read( streams=g.SG4[0] )
				
						# real and imaginary parts of the state concatenated into a unified tensor
						gate_01_real_st = gate_kernels_broadcasted_real_shared_dict[f"{gate_idx}_01"].read( streams=g.SG4[1])#, time=5 )
						gate_01_imag_st = gate_kernels_broadcasted_imag_shared_dict[f"{gate_idx}_01"].read( streams=g.SG4[1])#, time=5 )
						gate_01_st 	= g.stack( [gate_01_real_st, gate_01_imag_st], dim=0 )

						#gate_00_real_st = gate_kernels_broadcasted_real_shared_dict["0_00"].read( streams=g.SG4[6] )
						gate_01_st = g.reinterpret( gate_01_st, g.uint32 )
						gate_01_st = g.bitwise_and( gate_01_st, state_0_broadcasted_st, alus=[14], output_streams=g.SG4[0] )
						gate_01_st = g.reinterpret( gate_01_st, g.float32 )

						# real and imaginary parts of the state concatenated into a unified tensor
						gate_11_real_st = gate_kernels_broadcasted_real_shared_copy_dict[f"{gate_idx}_10"].read( streams=g.SG4[3], time=7 )
						gate_11_imag_st = gate_kernels_broadcasted_imag_shared_copy_dict[f"{gate_idx}_10"].read( streams=g.SG4[3], time=7 )
						gate_11_st 	= g.stack( [gate_11_real_st, gate_11_imag_st], dim=0 )

						gate_11_st = g.reinterpret( gate_11_st, g.uint32 )
						gate_11_st = g.bitwise_and( gate_11_st, state_1_broadcasted_st, alus=[6], output_streams=g.SG4[2] )
						gate_11_st = g.reinterpret( gate_11_st, g.float32 )
					
						# combine the filtered gate element vectors
						gate_second_column_st = g.add(gate_01_st, gate_11_st, alus=[3], output_streams=g.SG4[3])
				
							
					
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
					gate_second_column_imag_st                      = gate_second_column_imag_mt_256.read( streams=[g.SG4_W[5]] )
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
		

					layout_transformed_state_real_copy = layout_State_input_real_copy
					layout_transformed_state_imag_copy = layout_State_input_imag_copy

					print( g.vxm_identity.__doc__ )

					# calculate (Psi*gate_first_column).real + (Psi_permuted*gate_second_column).real
					Psi_transformed_real_st	= g.add( state__gate_first_column_real_st,  state_permuted__gate_second_column_real_st, alus=[7], output_streams=g.SG4_W[2] )
					Psi_transformed_real_st = g.vxm_identity( Psi_transformed_real_st, alus=[6], output_streams=g.SG4_W[2] )

					Psi_transformed_real_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real_{gate_idx}", storage_req=self.state_real_request, program_output=True)
					Psi_transformed_real_copy_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real_copy_{gate_idx}", storage_req=self.state_real_copy_request)

					g.add_mem_constraints(Psi_transformed_real_list, [Psi_transformed_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					Psi_transformed_real_list.append( Psi_transformed_real_mt )
				
					g.add_mem_constraints(Psi_transformed_real_copy_list, [Psi_transformed_real_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					Psi_transformed_real_copy_list.append( Psi_transformed_real_copy_mt )

					# calculate (Psi*gate_first_column).imag + (Psi_permuted*gate_second_column).imag
					Psi_transformed_imag_st	    = g.add( state__gate_first_column_imag_st,  state_permuted__gate_second_column_imag_st, alus=[14], output_streams=g.SG4_E[6] )
					Psi_transformed_imag_st = g.vxm_identity( Psi_transformed_imag_st, alus=[15], output_streams=g.SG4_W[0] )


					Psi_transformed_imag_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag_{gate_idx}", storage_req=self.state_imag_request, program_output=True)
					Psi_transformed_imag_copy_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag_copy_{gate_idx}", storage_req=self.state_imag_copy_request)
					
					g.add_mem_constraints(Psi_transformed_imag_list, [Psi_transformed_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					Psi_transformed_imag_list.append( Psi_transformed_imag_mt )
					
					g.add_mem_constraints(Psi_transformed_imag_copy_list, [Psi_transformed_imag_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
					Psi_transformed_imag_copy_list.append( Psi_transformed_imag_copy_mt )
					
					
			self.pgm_pkg.compile_program_context(pgm3)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
