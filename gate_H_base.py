from gate_counters import Gate_counters
from memory_layouts import *
import groq.api as g
import numpy as np

	
	

class Gate_H_base(Gate_counters):
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
		
		#self.target_qubit = 9

		self.transposer_request_to_qubit_state_readout  = g.tensor.create_transposer_request( [0], 1) # request transposer on the west side
		self.transposer_request_to_qubit_state_write    = g.tensor.create_transposer_request( [3], 1) # request transposer on the east side
		self.distributor_request_to_qubit_state_readout = g.tensor.create_distributor_request( [0], 1 )  # West side	
		self.distributor_request_to_qubit_state_write   = g.tensor.create_distributor_request( [6], 1 )  # East side	



	def select_target_qbit_read_map_west(self):
		"""
		Function used to read out the correct target qubit map to gather the target qubits L from the memory
		The target qubit map is used to select those memory addresses of the state vector for which the control qubit is in state 1.
		The target memory maps of control qubits 9, ..., 17 are stored in the lower half of one slice (addresses 0,1,..., 2^12).
		The target memory maps of control qubits 18, ..., 20 are stored in the upper half of one slice (addresses 2^12,..., 2^13). 		
		The 320-byte vector is splitted into 20 16-byte superlanes. 
		In each 16-byte segments the first two bytes are used to address a memory on a slice. 
		Thus, 8 memory maps can be stored in a single 16-byte superlane. Distributors are used to move the memory addresses into the position of the first two bytes.
		 
       
		Parameters
		----------

		Return
		------
		Returns with the stream tensor of the target qubits memory map.
		"""


		target_qubit_0_state_read_maps_list = self.target_qubit_read_maps_L_mt.split( dim=0, num_splits=memory_addresses)
		target_qubit_1_state_read_maps_list = self.target_qubit_read_pair_maps_L_mt.split( dim=0, num_splits=memory_addresses)
		read_out_state_0_memory_tensor_list = []
		read_out_state_1_memory_tensor_list = []
		for idx in range(memory_addresses//4):

			# idx, idx+1 needed for bank alternation strategy
			read_out_state_0_memory_tensor_list.append( target_qubit_0_state_read_maps_list[ memory_addresses//2 + 2*idx ] )	# for upper row of the gate kernel (output state is 0)
			read_out_state_0_memory_tensor_list.append( target_qubit_0_state_read_maps_list[ memory_addresses//2 + 2*idx+1 ] )	# for upper row of the gate kernel (output state is 0)
			read_out_state_0_memory_tensor_list.append( target_qubit_0_state_read_maps_list[ memory_addresses//2 + 2*idx ] )	# for lower row of the gate kernel (output state is 1) 
			read_out_state_0_memory_tensor_list.append( target_qubit_0_state_read_maps_list[ memory_addresses//2 + 2*idx+1 ] )	# for lower row of the gate kernel (output state is 1) 

			# idx, idx+1 needed for bank alternation strategy
			read_out_state_1_memory_tensor_list.append( target_qubit_1_state_read_maps_list[ memory_addresses//2 + 2*idx ] )	# for upper row of the gate kernel (output state is 0)
			read_out_state_1_memory_tensor_list.append( target_qubit_1_state_read_maps_list[ memory_addresses//2 + 2*idx+1 ] )	# for upper row of the gate kernel (output state is 0)
			read_out_state_1_memory_tensor_list.append( target_qubit_1_state_read_maps_list[ memory_addresses//2 + 2*idx ] )	# for lower row of the gate kernel (output state is 1) 
			read_out_state_1_memory_tensor_list.append( target_qubit_1_state_read_maps_list[ memory_addresses//2 + 2*idx+1 ] )	# for lower row of the gate kernel (output state is 1) 


		read_out_qubit_0_memory_tensor_mt = g.stack( read_out_state_0_memory_tensor_list, dim=0 )
		read_out_qubit_0_memory_tensor_st = read_out_qubit_0_memory_tensor_mt.read( streams=g.SG1[0] )
		read_out_qubit_0_memory_tensor_st = g.reshape( read_out_qubit_0_memory_tensor_st, [memory_addresses,320])


		read_out_qubit_1_memory_tensor_mt = g.stack( read_out_state_1_memory_tensor_list, dim=0 )
		read_out_qubit_1_memory_tensor_mt.set_program_output()
		read_out_qubit_1_memory_tensor_mt.name = "rrrrwwww1"
		read_out_qubit_1_memory_tensor_st = read_out_qubit_1_memory_tensor_mt.read( streams=g.SG1[1] )
		read_out_qubit_1_memory_tensor_st = g.reshape( read_out_qubit_1_memory_tensor_st, [memory_addresses,320])


		read_out_qubit_memory_tensor_st = g.stack( [read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st], dim=1 )
		
		
		gate_counter_distribute_map_selector_st = self.gate_counter_target_qubit_L_distributor_maps_write_mt.read( streams=[g.SG1_W[12]] )		
		distributor_map_st                      = g.mem_gather(self.target_qubit_distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[12]] )
		#distributor_map_st                      = self.target_qubit_distributor_map_mt.read( streams=g.SG1[6] )
		
		read_out_qubit_memory_tensor_st = g.distribute_8( read_out_qubit_memory_tensor_st, 
									distributor_map_st,
									distributor_req=self.distributor_request_to_qubit_state_readout,
									map_stream_req=g.SG1[6],
									bypass8=0b11111100,
									)	

		
		read_out_qubit_memory_tensor_st = g.transpose_null(read_out_qubit_memory_tensor_st, 
					                             transposer_req=self.transposer_request_to_qubit_state_readout, 
					                             stream_order=[0,1])


		read_out_qubit_memory_tensor_st_list = read_out_qubit_memory_tensor_st.split( dim=1, num_splits=2)	
		
		read_out_qubit_0_memory_tensor_st = g.reshape( read_out_qubit_memory_tensor_st_list[0], [memory_addresses,320])
		read_out_qubit_1_memory_tensor_st = g.reshape( read_out_qubit_memory_tensor_st_list[1], [memory_addresses,320])


		#read_out_qubit_0_memory_tensor_st.write( name="rrww0", layout='H1(W), -1, S1', program_output=True )
		#read_out_qubit_1_memory_tensor_st.write( name="rrww1", layout='H1(W), -1, S1(43)', program_output=True )
		#distributor_map_st.write( name="distr_map", layout='H1(W), -1, S1(43)', program_output=True )
		#self.target_qubit_distributor_map_mt.set_program_output()
		

		return read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st		




	def select_target_qbit_write_map_east(self):
		"""
		Function used to read out the correct target qubit map for scattering the target qubits L into the memory
		The target qubit map is used to select those memory addresses of the state vector for which the control qubit is in state 1.
		The target memory maps of control qubits 9, ..., 17 are stored in the lower half of one slice (addresses 0,1,..., 2^12).
		The target memory maps of control qubits 18, ..., 20 are stored in the upper half of one slice (addresses 2^12,..., 2^13). 		
		The 320-byte vector is splitted into 20 16-byte superlanes. 
		In each 16-byte segments the first two bytes are used to address a memory on a slice. 
		Thus, 8 memory maps can be stored in a single 16-byte superlane. Distributors are used to move the memory addresses into the position of the first two bytes.
		 
       
		Parameters
		----------

		Return
		------
		Returns with the stream tensor of the target qubits memory map.
		"""


		target_qubit_write_maps_H_st = self.target_qubit_write_maps_H_mt.read( streams=g.SG1[16] )

		
		
		
		gate_counter_distribute_map_selector_st = self.gate_counter_target_qubit_L_distributor_maps_write_mt.read( streams=[g.SG1_W[12]] )		
		distributor_map_st                      = g.mem_gather(self.target_qubit_distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[12]] )
		#distributor_map_st                      = self.target_qubit_distributor_map_mt.read( streams=g.SG1[8] )
		
		target_qubit_write_maps_H_st = g.distribute_8( target_qubit_write_maps_H_st, 
									distributor_map_st,
									distributor_req=self.distributor_request_to_qubit_state_write,
									map_stream_req=g.SG1[8],
									bypass8=0b11111110,
									)	

		
		target_qubit_write_maps_H_st = g.transpose_null(target_qubit_write_maps_H_st, 
					                             transposer_req=self.transposer_request_to_qubit_state_write, 
					                             stream_order=[0])	

		#rr = target_qubit_write_maps_L_st.write(name="uuu", layout='H1(W), -1, S1(43)', program_output=True )

		return target_qubit_write_maps_H_st	

	
	def two_complex_dot_product(self, s_r_0, s_i_0, s_r_1, s_i_1, g_r_0, g_i_0, g_r_1, g_i_1):
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
		r_st_m1 = g.add(r0_st_m2, r1_st_m2, alus=[3], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[1], g.SG4_E[0]], direction=Direction.WESTWARD
		i_st_m1 = g.add(i0_st_m2, i1_st_m2, alus=[6], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[3], g.SG4_E[4]], direction=Direction.WESTWARD
		r_delay_st_p0 = g.vxm_identity(r_st_m1, time=0, alus=[13], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[0]], direction=Direction.WESTWARD
		i_delay_st_p0 = g.vxm_identity(i_st_m1, alus=[5], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[2]], direction=Direction.WESTWARD
		return r_delay_st_p0, i_delay_st_p0 #directions=[Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD]
	

	def read_out_gate_elements( self, element_num ):
	
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
	
	
		gate_element_counter_first_column_list  = []
		gate_element_counter_second_column_list = []
		for idx in range(element_num//4):
			gate_element_counter_first_column_list.append( self.gate_counter_gate_element_selector_00_mt)
			gate_element_counter_first_column_list.append( self.gate_counter_gate_element_selector_00_mt )
			gate_element_counter_first_column_list.append( self.gate_counter_gate_element_selector_10_mt )
			gate_element_counter_first_column_list.append( self.gate_counter_gate_element_selector_10_mt )	

			gate_element_counter_second_column_list.append( self.gate_counter_gate_element_selector_01_mt )
			gate_element_counter_second_column_list.append( self.gate_counter_gate_element_selector_01_mt )
			gate_element_counter_second_column_list.append( self.gate_counter_gate_element_selector_11_mt )
			gate_element_counter_second_column_list.append( self.gate_counter_gate_element_selector_11_mt )	
					
					
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
		
	def read_out_state_2_transform( self, read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st):
	
		"""
		Function used to read out the state vector to be transfromed according to the predefined memory maps
		
		
		Parameters
		----------
		read_out_qubit_0_memory_tensor_st : StreamTensor
			The memory map to read out the state vector elements standing for target qbit state 0
		read_out_qubit_1_memory_tensor_st : StreamTensor
			The memory map to read out the state vector elements standing for target qbit state 1
		
		Return
		------
		Returns with the stream tensors of the state vector (real and imaginary parts)
		
		Psi_transformed_real_state_0 : StreamTensor
			The real part of the input state vector, where the target qubit is in state 0
		Psi_transformed_imag_state_0 : StreamTensor
			The imaginary part of the input state vector, where the target qubit is in state 0
		Psi_transformed_real_state_1 : StreamTensor
			The real part of the input state vector, where the target qubit is in state 1
		Psi_transformed_imag_state_1 : StreamTensor
			The imaginary part of the input state vector, where the target qubit is in state 1
		
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
					
		element_num = read_out_qubit_1_memory_tensor_st.shape[0]
				


		Psi_transformed_real_state_0_bytes_list = []
		Psi_transformed_imag_state_0_bytes_list = []
		Psi_transformed_real_state_0_copy_bytes_list = []
		Psi_transformed_imag_state_0_copy_bytes_list = []
		#gate_element_real_copy_list = []
		#gate_element_imag_copy_list = []
		for byte_idx in range(4):
					
			################
			output_stream_offset = 4*4
			Psi_transformed_real_state_0_byte_st = g.mem_gather( Psi_real_bytes_mt_list[byte_idx], read_out_qubit_0_memory_tensor_st,
										 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
			Psi_transformed_real_state_0_byte_st = g.reshape( Psi_transformed_real_state_0_byte_st, [element_num,1,320])
			Psi_transformed_real_state_0_bytes_list.append( Psi_transformed_real_state_0_byte_st )

			################
			output_stream_offset = 1*4
			Psi_transformed_imag_state_0_byte_st = g.mem_gather( Psi_imag_bytes_mt_list[byte_idx], read_out_qubit_0_memory_tensor_st,
												 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
			Psi_transformed_imag_state_0_byte_st = g.reshape( Psi_transformed_imag_state_0_byte_st, [element_num,1,320])
			Psi_transformed_imag_state_0_bytes_list.append( Psi_transformed_imag_state_0_byte_st )

			################
			output_stream_offset = 5*4
			Psi_transformed_real_state_0_copy_byte_st = g.mem_gather( Psi_real_bytes_copy_mt_list[byte_idx], read_out_qubit_1_memory_tensor_st,
												 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
			Psi_transformed_real_state_0_copy_byte_st = g.reshape( Psi_transformed_real_state_0_copy_byte_st, [element_num,1,320])
			Psi_transformed_real_state_0_copy_bytes_list.append( Psi_transformed_real_state_0_copy_byte_st )


			################
			output_stream_offset = 7*4
			Psi_transformed_imag_state_0_copy_byte_st = g.mem_gather( Psi_imag_bytes_copy_mt_list[byte_idx], read_out_qubit_1_memory_tensor_st,
												 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
			Psi_transformed_imag_state_0_copy_byte_st = g.reshape( Psi_transformed_imag_state_0_copy_byte_st, [element_num,1,320])
			Psi_transformed_imag_state_0_copy_bytes_list.append( Psi_transformed_imag_state_0_copy_byte_st )

					
					
		Psi_transformed_real_state_0      = g.concat( Psi_transformed_real_state_0_bytes_list, dim=1 )
		Psi_transformed_imag_state_0      = g.concat( Psi_transformed_imag_state_0_bytes_list, dim=1 )
		Psi_transformed_real_state_1      = g.concat( Psi_transformed_real_state_0_copy_bytes_list, dim=1 )
		Psi_transformed_imag_state_1      = g.concat( Psi_transformed_imag_state_0_copy_bytes_list, dim=1 )
							
		Psi_transformed_real_state_0      = g.reinterpret(Psi_transformed_real_state_0, dtype=g.float32 )
		Psi_transformed_imag_state_0      = g.reinterpret(Psi_transformed_imag_state_0, dtype=g.float32 )
		Psi_transformed_real_state_1      = g.reinterpret(Psi_transformed_real_state_1, dtype=g.float32 )
		Psi_transformed_imag_state_1      = g.reinterpret(Psi_transformed_imag_state_1, dtype=g.float32 )
		
		
		return Psi_transformed_real_state_0, Psi_transformed_imag_state_0, Psi_transformed_real_state_1, Psi_transformed_imag_state_1
		
		
		
		
	def write_tranformed_states_2_mem( self, Psi_transformed_real_st, Psi_transformed_imag_st, target_qubit_L_write_map_st):
	
		"""
		Function used to scatter the transformed state vector into the memory
		
		
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
		Psi_transformed_real_mt_320_8       = g.reinterpret(Psi_transformed_real_st, dtype=g.int8)
		Psi_transformed_imag_mt_320_8       = g.reinterpret(Psi_transformed_imag_st, dtype=g.int8)			
		Psi_transformed_real_bytes_mt_list  = g.split(Psi_transformed_real_mt_320_8, dim=-2, num_splits=g.float32.size)
		Psi_transformed_imag_bytes_mt_list  = g.split(Psi_transformed_imag_mt_320_8, dim=-2, num_splits=g.float32.size)		
				
		for byte_idx in range(4):
					
			g.mem_scatter( Psi_transformed_real_bytes_mt_list[byte_idx], Psi_real_bytes_mt_list[byte_idx], target_qubit_L_write_map_st)
			g.mem_scatter( Psi_transformed_real_bytes_mt_list[byte_idx], Psi_real_bytes_copy_mt_list[byte_idx], target_qubit_L_write_map_st)
			g.mem_scatter( Psi_transformed_imag_bytes_mt_list[byte_idx], Psi_imag_bytes_mt_list[byte_idx], target_qubit_L_write_map_st)
			g.mem_scatter( Psi_transformed_imag_bytes_mt_list[byte_idx], Psi_imag_bytes_copy_mt_list[byte_idx], target_qubit_L_write_map_st)
					
					
					

		Psi_transformed_real_mt_output = g.constant_tensor( self.State_input_real_mt.shape, g.float32, storage_req=self.State_input_real_mt.storage_request, 
                                                                                    name=f"Psi_transformed_real", initialized=False)
		Psi_transformed_imag_mt_output = g.constant_tensor( self.State_input_imag_mt.shape, g.float32, storage_req=self.State_input_imag_mt.storage_request, 
                                                                                    name=f"Psi_transformed_imag", initialized=False)

		#Psi_transformed_real_mt_output.set_program_output()
		#Psi_transformed_imag_mt_output.set_program_output()

