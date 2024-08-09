from memory_layouts import *
import groq.api as g
import numpy as np
from groq.api import instruction as inst

class Upload:
	def __init__(self, pgm_pkg, control_qbit):
		self.control_qbit = control_qbit
		self.pgm_pkg = pgm_pkg
		self.gate_kernels_real_packed_mt = None
		self.gate_kernels_imag_packed_mt = None
		self.State_input_real_mt = None
		self.State_input_imag_mt = None
		self.State_input_real_copy_mt = None
		self.State_input_imag_copy_mt = None
		self.state_permute_map_selector_mt = None
		self.permute_maps_west = None
		self.permute_maps_east = None
		self.state_1_broadcasted_mt = None
		self.state_0_broadcasted_mt = None
		self.control_state_1_broadcasted_mt = None
		self.control_state_0_broadcasted_mt = None
		self.ones_mt = None
		self.zeros_mt = None

		# create reusable memory requests
		self.state_imag_request = g.tensor.create_storage_request( layout=layout_State_input_imag )
		self.state_real_request = g.tensor.create_storage_request( layout=layout_State_input_real )

		self.state_imag_copy_request = g.tensor.create_storage_request( layout=layout_State_input_imag_copy )
		self.state_real_copy_request = g.tensor.create_storage_request( layout=layout_State_input_real_copy )

	def upload_program(self):
		# Defines a program to upload the inputs
		name = "upload"
		with self.pgm_pkg.create_program_context(name) as pgm1:

			self.State_input_real_mt = g.input_tensor(shape=(memory_addresses, matrix_size,), dtype=g.float32, name="State_input_real", storage_req=self.state_real_request)
			self.State_input_imag_mt = g.input_tensor(shape=(memory_addresses, matrix_size,), dtype=g.float32, name="State_input_imag", storage_req=self.state_imag_request)

			self.State_input_real_copy_mt = g.input_tensor(shape=(memory_addresses, matrix_size,), dtype=g.float32, name="State_input_real_copy", storage_req=self.state_real_copy_request)
			self.State_input_imag_copy_mt = g.input_tensor(shape=(memory_addresses, matrix_size,), dtype=g.float32, name="State_input_imag_copy", storage_req=self.state_imag_copy_request)
							
			
			self.state_permute_map_selector_mt 	= g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="state_permute_map_selector", layout=layout_state_permute_map_selector)
			state_1_selector_mt				= g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="state_1_selector", layout=layout_state_1_selector)
			state_0_selector_mt				= g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="state_0_selector", layout=layout_state_0_selector)

			control_state_1_selector_mt = g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="control_state_1_selector", layout=layout_control_state_1_selector)
			control_state_0_selector_mt = g.input_tensor(shape=(gate_count,320,), dtype=g.uint8, name="control_state_0_selector", layout=layout_control_state_0_selector)

			self.gate_kernels_real_packed_mt = g.input_tensor(shape=(320,), dtype=g.float32, name="gate_kernels_real_packed", layout=layout_gate_kernels_real_packed)
			self.gate_kernels_imag_packed_mt = g.input_tensor(shape=(320,), dtype=g.float32, name="gate_kernels_imag_packed", layout=layout_gate_kernels_imag_packed)


			self.State_input_real_mt.is_static 			= True
			self.State_input_imag_mt.is_static 			= True
			self.State_input_real_copy_mt.is_static 			= True
			self.State_input_imag_copy_mt.is_static 			= True
			self.state_permute_map_selector_mt.is_static = True
			self.gate_kernels_real_packed_mt.is_static 	= True
			self.gate_kernels_imag_packed_mt.is_static 	= True
			state_1_selector_mt.is_static			= True
			state_0_selector_mt.is_static			= True
			control_state_1_selector_mt.is_static = True
			control_state_0_selector_mt.is_static = True

			# generate permutation maps to reorder the state vector elements for qubits less than small_qbit_num_limit
			permute_map = np.zeros( (small_qbit_num_limit,320,), dtype=np.uint32 )

			for target_qbit_loc in range(small_qbit_num_limit):

				permute_map_np = np.zeros( (256,), dtype=np.uint32 )
				target_qbit_pair_diff = 1 << target_qbit_loc
				print('target_qubit_pair_diff', target_qbit_pair_diff)
				
				for idx in range(256): # 256 = 2^8
					if (idx < matrix_size):
						permute_map_np[idx] = idx ^ target_qbit_pair_diff
					else:
						permute_map_np[idx] = idx
				

				permute_map[target_qbit_loc,:] = inst.encode_permute_map(  permute_map_np.tolist() )
				print(' ')
				print(permute_map[target_qbit_loc,:])

			self.permute_maps_mt_west = g.from_data( np.asarray(permute_map, dtype=np.uint8), layout=layout_permute_maps_west )
			self.permute_maps_mt_east = g.from_data( np.asarray(permute_map, dtype=np.uint8), layout=layout_permute_maps_east )
			self.permute_maps_mt_west.is_static = True
			self.permute_maps_mt_east.is_static = True
			print(self.permute_maps_mt_west.shape)
			print(dir(self.permute_maps_mt_west))
			
			
			# generate vectors to indicate which elements in a 320 vector correspond to state |1> of target qubit less than small_qbit_num_limit 
			# if target_qubit_loc at the idx-th element is in state_1 |1> then all the 8 bits of the idx-th element is set to 1, thus state_1[idx] = 255
			# if target_qubit_loc at the idx-th element is in state_0 |0> then all the 8 bits of the idx-th element is set to 1, thus state_0[idx] = 255
			states_1_np = np.zeros( (small_qbit_num_limit,320,), dtype=np.uint8 )
			states_0_np = np.zeros( (small_qbit_num_limit,320,), dtype=np.uint8 )
			for target_qbit_loc in range(small_qbit_num_limit):
				target_qbit_pair_diff = 1 << target_qbit_loc
				for idx in range(256): # 256 = 2^8
					if (idx & target_qbit_pair_diff > 0):
						states_1_np[target_qbit_loc, idx] = 255
						states_0_np[target_qbit_loc, idx] = 0
					else:
						states_1_np[target_qbit_loc, idx] = 0
						states_0_np[target_qbit_loc, idx] = 255
				print( "states |1> at target qubit ", target_qbit_loc, " is:" )
				print( states_0_np[target_qbit_loc, 0:256] )

			states_1_mt = g.from_data( np.asarray(states_1_np, dtype=np.uint8), layout=layout_states_1 )
			states_1_mt.is_static = True

			states_0_mt = g.from_data( np.asarray(states_0_np, dtype=np.uint8), layout=layout_states_0 )
			states_0_mt.is_static = True


			# create reusable distributor requests
			distributor_requests = []
			distributor_requests.append(  g.tensor.create_distributor_request( [4], 1 ) ) # EAST
			distributor_requests.append(  g.tensor.create_distributor_request( [7], 1 ) ) # EAST	

			# create reusable transposer requests
			transposer_requests = []
			transposer_requests.append(  g.tensor.create_transposer_request( [2], 1 ) ) # EAST
			transposer_requests.append(  g.tensor.create_transposer_request( [3], 1 ) ) # EAST	

			# distribute maps to broadcast teh gate kernel elements over a vector
			distribute_map_np = np.zeros( (320,), dtype=np.uint8 )
			for idx in range(320):
				distribute_map_np[idx] = idx % 16

			distribute_map_tensor_state1_mt = g.from_data( np.asarray(distribute_map_np, dtype=np.uint8), layout=layout_distribute_map_tensor_state1 + f", A1({small_qbit_num_limit+4})" )
			distribute_map_tensor_state0_mt = g.from_data( np.asarray(distribute_map_np, dtype=np.uint8), layout=layout_distribute_map_tensor_state0 + f", A1({small_qbit_num_limit+4})" )
			g.add_mem_constraints([states_1_mt], [distribute_map_tensor_state1_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			g.add_mem_constraints([states_0_mt], [distribute_map_tensor_state0_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			g.add_mem_constraints([state_1_selector_mt], [control_state_1_selector_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
			g.add_mem_constraints([state_0_selector_mt], [control_state_0_selector_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)

			# generate tensors full with float32 ones and zeros
			self.ones_mt = g.full((1, 1, 320), 1.0, dtype=g.float32, name="ones_mt", layout=layout_ones)
			self.zeros_mt = g.full((1, 1, 320), 0.0, dtype=g.float32, name="zeros_mt", layout=layout_zeros)
			self.ones_mt = g.reinterpret( self.ones_mt, dtype=g.uint32)
			self.zeros_mt = g.reinterpret( self.zeros_mt, dtype=g.uint32)
			self.ones_mt.is_static = True
			self.zeros_mt.is_static = True

			with g.ResourceScope(name="distribute_state_indices_scope", is_buffered=True, time=0) as distribute_state_indices_scope :

				#state_1_st = state_1_mt.read( streams=g.SG1[0], time=0 )
				#state_0_st = state_0_mt.read( streams=g.SG1[24], time=0 )

				state_1_selector_st = state_1_selector_mt.read(streams=[g.SG1[0]])
				state_1_st = g.mem_gather(states_1_mt, state_1_selector_st, output_streams=[g.SG1[0]], time=0)

				state_0_selector_st = state_0_selector_mt.read(streams=[g.SG1[24]])
				state_0_st = g.mem_gather(states_0_mt, state_0_selector_st, output_streams=[g.SG1[24]], time=0)


				distributor_request = distributor_requests[0]
				state_1_broadcasted_st = g.distribute_lowest( state_1_st, 
						distribute_map_tensor_state1_mt,
						distributor_req=distributor_request,
						map_stream_req=g.SG1[12],
						bypass8=0b00001111,
				)

				state_1_broadcasted_st = g.transpose_null(
					state_1_broadcasted_st,
					transposer_req=transposer_requests[0],
					stream_order=[4,5,6,7],
				)

		
				distributor_request = distributor_requests[1]
				state_0_broadcasted_st = g.distribute_lowest( state_0_st, 
						distribute_map_tensor_state0_mt,
						distributor_req=distributor_request,
						map_stream_req=g.SG1[31],
						bypass8=0b00001111,
				)

				state_0_broadcasted_st = g.transpose_null(
					state_0_broadcasted_st,
					transposer_req=transposer_requests[1],
					stream_order=[12,13,14,15],
				)

		
				state_1_broadcasted_st = g.reinterpret( state_1_broadcasted_st, g.uint32 )
				self.state_1_broadcasted_mt = state_1_broadcasted_st.write(name="state_1_broadcasted", layout=layout_state_1_broadcasted , program_output=False)
				state_0_broadcasted_st = g.reinterpret( state_0_broadcasted_st, g.uint32 )
				self.state_0_broadcasted_mt = state_0_broadcasted_st.write(name="state_0_broadcasted", layout=layout_state_0_broadcasted, program_output=False)

				self.state_1_broadcasted_mt.is_static = True
				self.state_0_broadcasted_mt.is_static = True

				# same as before but now for the control qbit
				control_state_1_selector_st = control_state_1_selector_mt.read(streams=[g.SG1[0]])
				control_state_1_st = g.mem_gather(states_1_mt, control_state_1_selector_st, output_streams=[g.SG1[0]], time=3)
				
				control_state_0_selector_st = control_state_0_selector_mt.read(streams=[g.SG1[24]])
				control_state_0_st = g.mem_gather(states_0_mt, control_state_0_selector_st, output_streams=[g.SG1[24]], time=3)
				
				
				distributor_request = distributor_requests[0]
				control_state_1_broadcasted_st = g.distribute_lowest( control_state_1_st, 
						distribute_map_tensor_state1_mt,
						distributor_req=distributor_request,
						map_stream_req=g.SG1[12],
						bypass8=0b00001111,
				)
				
				control_state_1_broadcasted_st = g.transpose_null(
					control_state_1_broadcasted_st,
					transposer_req=transposer_requests[0],
					stream_order=[4,5,6,7],
				)
				
					
				distributor_request = distributor_requests[1]
				control_state_0_broadcasted_st = g.distribute_lowest( control_state_0_st, 
						distribute_map_tensor_state0_mt,
						distributor_req=distributor_request,
						map_stream_req=g.SG1[31],
						bypass8=0b00001111,
				)
				
				control_state_0_broadcasted_st = g.transpose_null(
					control_state_0_broadcasted_st,
					transposer_req=transposer_requests[1],
					stream_order=[12,13,14,15],
				)
				
				control_state_0_broadcasted_st = g.reinterpret( control_state_0_broadcasted_st, g.uint32 )
				control_state_1_broadcasted_st = g.reinterpret( control_state_1_broadcasted_st, g.uint32 )
				self.control_state_1_broadcasted_mt = control_state_1_broadcasted_st.write(name="control_state_1_broadcasted", layout=layout_control_state_1_broadcasted, program_output=False)
				self.control_state_0_broadcasted_mt = control_state_0_broadcasted_st.write(name="control_state_0_broadcasted", layout=layout_control_state_0_broadcasted, program_output=False)
				
				self.control_state_1_broadcasted_mt.is_static = True
				self.control_state_0_broadcasted_mt.is_static = True
				g.add_mem_constraints([self.control_state_0_broadcasted_mt], [self.state_0_broadcasted_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.control_state_1_broadcasted_mt], [self.state_1_broadcasted_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
				g.add_mem_constraints([self.state_0_broadcasted_mt, self.control_state_0_broadcasted_mt], [self.zeros_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)

			
			self.pgm_pkg.compile_program_context(pgm1)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)
		#############################################################################################################x
		# program to split the gate kernels and other metada between the EAST-WAST hemispheres 
