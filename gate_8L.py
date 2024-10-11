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
		r_st = g.add(r0_st_m2, r1_st_m2, alus=[3], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[1], g.SG4_E[0]], direction=Direction.WESTWARD
		i_st = g.add(i0_st_m2, i1_st_m2, alus=[6], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[3], g.SG4_E[4]], direction=Direction.WESTWARD
		r_delay_st_p0 = g.vxm_identity(r_st, alus=[13], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[0]], direction=Direction.WESTWARD
		i_delay_st_p0 = g.vxm_identity(i_st, alus=[5], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[2]], direction=Direction.WESTWARD
		return r_st, i_st, r_delay_st_p0, i_delay_st_p0 #directions=[Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD]


	def select_target_qbit_read_map_west_8L(self):
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

		read_out_qubit_0_memory_tensor_st = self.target_qubit_read_maps_8L_mt.read( streams=g.SG1[0] )
		read_out_qubit_1_memory_tensor_st = self.target_qubit_read_maps_8L_pair_mt.read( streams=g.SG1[1] )

		
		read_out_qubit_memory_tensor_st = g.stack( [read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st], dim=1 )
		
		
		gate_counter_distribute_map_selector_st = self.gate_counter_distributor_maps_west_mt.read( streams=[g.SG1[12]] )		
		distributor_map_st                      = g.mem_gather(self.distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[12]] )
		#distributor_map_st                      = self.distributor_map_mt.read( streams=g.SG1[6] )
		
		read_out_qubit_memory_tensor_st = g.distribute_8( read_out_qubit_memory_tensor_st, 
									distributor_map_st,
									distributor_req=self.distributor_request_to_qubit_state_readout,
									map_stream_req=g.SG1[12],
									bypass8=0b11111100,
									)	

		
		read_out_qubit_memory_tensor_st = g.transpose_null(read_out_qubit_memory_tensor_st, 
					                             transposer_req=self.transposer_request_to_qubit_state_readout, 
					                             stream_order=[0,1])


		read_out_qubit_memory_tensor_st_list = read_out_qubit_memory_tensor_st.split( dim=1, num_splits=2)	
		
		read_out_qubit_0_memory_tensor_st = g.reshape( read_out_qubit_memory_tensor_st_list[0], [memory_addresses//2,320])
		read_out_qubit_1_memory_tensor_st = g.reshape( read_out_qubit_memory_tensor_st_list[1], [memory_addresses//2,320])
		

		#read_out_qubit_0_memory_tensor_st.write( name="rrww0", layout='H1(W), -1, S1(31)', program_output=True )
		#read_out_qubit_1_memory_tensor_st.write( name="rrww1", layout='H1(W), -1, S1(30)', program_output=True )
		#distributor_map_st.write( name="distr_map", layout='H1(W), -1, S1(43)', program_output=True )
		#self.target_qubit_distributor_map_mt.set_program_output()
		

		return read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st

	def select_target_qbit_write_map_east_8L(self):
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
		time_delay = 3

		write_qubit_memory_tensor_st_a = self.target_qubit_write_maps_8L_mt.read( streams=g.SG1[16], time=time_delay )
		write_qubit_memory_tensor_st_b = self.target_qubit_write_maps_8L_mt_b.read( streams=g.SG1[19], time=0 )

		write_qubit_memory_tensor_st_a_list = g.split( write_qubit_memory_tensor_st_a, dim=0, num_splits=memory_addresses//2 )
		write_qubit_memory_tensor_st_b_list = g.split( write_qubit_memory_tensor_st_b, dim=0, num_splits=memory_addresses//2 )


		write_qubit_memory_tensor_st_a_first_part = g.concat(write_qubit_memory_tensor_st_a_list[0:memory_addresses//2-time_delay], dim=0 )
		write_qubit_memory_tensor_st_a_second_part = g.concat(write_qubit_memory_tensor_st_a_list[memory_addresses//2-time_delay:], dim=0 )

		write_qubit_memory_tensor_st_b_first_part = g.concat(write_qubit_memory_tensor_st_b_list[0:time_delay], dim=0 )
		write_qubit_memory_tensor_st_b_second_part = g.concat(write_qubit_memory_tensor_st_b_list[time_delay:memory_addresses//2], dim=0 )


		
		##########################################	
		
		gate_counter_distribute_map_selector_st = self.gate_counter_distributor_maps_west_mt.read( streams=[g.SG1[11]] )		
		distributor_map_st                      = g.mem_gather(self.target_qubit_distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[10]] )
		#distributor_map_st                      = self.distributor_map_mt.read( streams=g.SG1[8] )
		
		write_qubit_memory_tensor_st_b_first_part = g.distribute_8( write_qubit_memory_tensor_st_b_first_part, 
									distributor_map_st,
									distributor_req=self.distributor_request_to_qubit_state_write,
									map_stream_req=g.SG1[8],
									bypass8=0b11110111,
									)	
		
		
		write_qubit_memory_tensor_st_b_first_part = g.transpose_null(write_qubit_memory_tensor_st_b_first_part, 
					                             transposer_req=self.transposer_request_to_qubit_state_write, 
					                             stream_order=[3])	

		write_qubit_memory_tensor_st_b_first_part = g.reshape( write_qubit_memory_tensor_st_b_first_part, [time_delay,320])


		##########################################
		write_qubit_memory_tensor_st = g.stack( [write_qubit_memory_tensor_st_a_first_part, write_qubit_memory_tensor_st_b_second_part], dim=1 )
	
			
		gate_counter_distribute_map_selector_st = self.gate_counter_distributor_maps_west_mt.read( streams=[g.SG1[11]] )		
		distributor_map_st                      = g.mem_gather(self.target_qubit_distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[10]] )
		#distributor_map_st                      = self.distributor_map_mt.read( streams=g.SG1[8] )
		
		write_qubit_memory_tensor_st = g.distribute_8( write_qubit_memory_tensor_st, 
									distributor_map_st,
									distributor_req=self.distributor_request_to_qubit_state_write,
									map_stream_req=g.SG1[8],
									bypass8=0b11110110,
									)	

		
		write_qubit_memory_tensor_st = g.transpose_null(write_qubit_memory_tensor_st, 
					                             transposer_req=self.transposer_request_to_qubit_state_write, 
					                             stream_order=[0,3])	
		

		write_qubit_memory_tensor_st_list = write_qubit_memory_tensor_st.split( dim=1, num_splits=2)

		
		write_qubit_memory_tensor_st_a_first_part  = g.reshape( write_qubit_memory_tensor_st_list[0], [(memory_addresses//2)-time_delay,320])
		write_qubit_memory_tensor_st_b_second_part = g.reshape( write_qubit_memory_tensor_st_list[1], [(memory_addresses//2)-time_delay,320])




		##########################################
		gate_counter_distribute_map_selector_st = self.gate_counter_distributor_maps_west_mt.read( streams=[g.SG1[11]] )		
		distributor_map_st                      = g.mem_gather(self.target_qubit_distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[10]] )
		#distributor_map_st                      = self.distributor_map_mt.read( streams=g.SG1[8] )
		
		write_qubit_memory_tensor_st_a_second_part = g.distribute_8( write_qubit_memory_tensor_st_a_second_part, 
									distributor_map_st,
									distributor_req=self.distributor_request_to_qubit_state_write,
									map_stream_req=g.SG1[8],
									bypass8=0b11111110,
									)	
		
		
		write_qubit_memory_tensor_st_a_second_part = g.transpose_null(write_qubit_memory_tensor_st_a_second_part, 
					                             transposer_req=self.transposer_request_to_qubit_state_write, 
					                             stream_order=[0])	


		write_qubit_memory_tensor_st_a_second_part = g.reshape( write_qubit_memory_tensor_st_a_second_part, [time_delay,320])
		
		write_qubit_memory_tensor_st_a = g.concat( [write_qubit_memory_tensor_st_a_first_part, write_qubit_memory_tensor_st_a_second_part], dim=0 )
		write_qubit_memory_tensor_st_b = g.concat( [write_qubit_memory_tensor_st_b_first_part, write_qubit_memory_tensor_st_b_second_part], dim=0 )

		print( 'ooooooooooooooooooo ', write_qubit_memory_tensor_st_a.shape )
		print( 'ooooooooooooooooooo ', write_qubit_memory_tensor_st_b.shape )

		#write_qubit_memory_tensor_st_a.write(name = 'hhhh', layout='H1(E), -1, S1', program_output=True )
		#write_qubit_memory_tensor_st_b.write(name = 'hhhh2', layout='H1(E), -1, S1', program_output=True )

		'''
		#write_qubit_memory_tensor_st = g.stack( [write_qubit_memory_tensor_st_a, write_qubit_memory_tensor_st_b], dim=1 )
		
		
		#gate_counter_distribute_map_selector_st = self.gate_counter_distributor_maps_west_mt.read( streams=[g.SG1_W[12]] )		
		#distributor_map_st                      = g.mem_gather(self.target_qubit_distributor_map_mt, gate_counter_distribute_map_selector_st, output_streams=[g.SG1[12]] )
		distributor_map_st                      = self.distributor_map_mt.read( streams=g.SG1[8] )
		
		write_qubit_memory_tensor_st_a = g.distribute_8( write_qubit_memory_tensor_st_a, 
									distributor_map_st,
									distributor_req=self.distributor_request_to_qubit_state_write,
									map_stream_req=g.SG1[8],
									bypass8=0b11111110,
									)	

		
		write_qubit_memory_tensor_st_a = g.transpose_null(write_qubit_memory_tensor_st_a, 
					                             transposer_req=self.transposer_request_to_qubit_state_write, 
					                             stream_order=[0])	


		
		#write_qubit_memory_tensor_mt_b = write_qubit_memory_tensor_st_a.write(name='write_qubit_memory_tensor_b', layout=layout_target_qubit_write_maps_8L_b)
		#write_qubit_memory_tensor_mt_c = write_qubit_memory_tensor_st_a.write(name='write_qubit_memory_tensor_c', layout=layout_target_qubit_write_maps_8L_c)

		#write_qubit_memory_tensor_st_a = write_qubit_memory_tensor_mt_b.read( streams=g.SG1[16] )
		#write_qubit_memory_tensor_st_b = write_qubit_memory_tensor_mt_c.read( streams=g.SG1[19] )
		'''
		'''
		write_qubit_memory_tensor_st_list = write_qubit_memory_tensor_st.split( dim=1, num_splits=2)	
		
		write_qubit_memory_tensor_st_a = g.reshape( write_qubit_memory_tensor_st_list[0], [memory_addresses//2,320])
		write_qubit_memory_tensor_st_b = g.reshape( write_qubit_memory_tensor_st_list[1], [memory_addresses//2,320])



		'''
		#rr = target_qubit_write_maps_L_st.write(name="uuu", layout='H1(W), -1, S1(43)', program_output=True )


		#write_qubit_memory_tensor_st_a = self.target_qubit_write_maps_8L_mt.read( streams=g.SG1[16] )
		#write_qubit_memory_tensor_st_b = self.target_qubit_write_maps_8L_mt_b.read( streams=g.SG1[19] )

		return write_qubit_memory_tensor_st_a, write_qubit_memory_tensor_st_b


	def read_out_state_2_transform_8L( self, read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st):
	
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

			Psi_real_bytes_mt      = Psi_real_bytes_mt_list[byte_idx]
			Psi_imag_bytes_mt      = Psi_imag_bytes_mt_list[byte_idx]
			Psi_real_bytes_copy_mt = Psi_real_bytes_copy_mt_list[byte_idx]
			Psi_imag_bytes_copy_mt = Psi_imag_bytes_copy_mt_list[byte_idx]
			'''
			Psi_real_bytes_mt_element_list = g.split( Psi_real_bytes_mt, dim=0, num_splits=memory_addresses )
			Psi_imag_bytes_mt_element_list = g.split( Psi_imag_bytes_mt, dim=0, num_splits=memory_addresses )
			Psi_real_bytes_copy_mt_element_list = g.split( Psi_real_bytes_copy_mt, dim=0, num_splits=memory_addresses )
			Psi_imag_bytes_copy_mt_element_list = g.split( Psi_imag_bytes_copy_mt, dim=0, num_splits=memory_addresses )
			'''
			read_out_qubit_0_memory_tensor_st_element_list = g.split( read_out_qubit_0_memory_tensor_st, dim=0, num_splits=element_num )
			read_out_qubit_1_memory_tensor_st_element_list = g.split( read_out_qubit_1_memory_tensor_st, dim=0, num_splits=element_num )

			Psi_transformed_real_state_0_byte_st_list = []
			Psi_transformed_imag_state_0_byte_st_list = []
			Psi_transformed_real_state_1_byte_st_list = []
			Psi_transformed_imag_state_1_byte_st_list = []

			for element_idx in range(element_num):

				if element_idx % 2 == 0:
					output_stream_offset = 4*4
					Psi_transformed_real_state_0_byte_st_list.append(
										g.mem_gather( Psi_real_bytes_mt, read_out_qubit_0_memory_tensor_st_element_list[element_idx],
										 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
										)

					output_stream_offset = 1*4
					Psi_transformed_imag_state_0_byte_st_list.append(
										g.mem_gather( Psi_imag_bytes_mt, read_out_qubit_0_memory_tensor_st_element_list[element_idx],
										 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
										)

					output_stream_offset = 5*4
					Psi_transformed_real_state_1_byte_st_list.append(
										g.mem_gather( Psi_real_bytes_copy_mt, read_out_qubit_1_memory_tensor_st_element_list[element_idx],
										 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
										)

					output_stream_offset = 7*4
					Psi_transformed_imag_state_1_byte_st_list.append(
										g.mem_gather( Psi_imag_bytes_copy_mt, read_out_qubit_1_memory_tensor_st_element_list[element_idx],
										 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
										)

				else:
					output_stream_offset = 4*4
					Psi_transformed_real_state_0_byte_st_list.append(
										g.mem_gather( Psi_real_bytes_copy_mt, read_out_qubit_0_memory_tensor_st_element_list[element_idx],
										 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
										)

					output_stream_offset = 1*4
					Psi_transformed_imag_state_0_byte_st_list.append(
										g.mem_gather( Psi_imag_bytes_copy_mt, read_out_qubit_0_memory_tensor_st_element_list[element_idx],
										 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
										)


					output_stream_offset = 5*4
					Psi_transformed_real_state_1_byte_st_list.append(
										g.mem_gather( Psi_real_bytes_mt, read_out_qubit_1_memory_tensor_st_element_list[element_idx],
										 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
										)

					output_stream_offset = 7*4
					Psi_transformed_imag_state_1_byte_st_list.append(
										g.mem_gather( Psi_imag_bytes_mt, read_out_qubit_1_memory_tensor_st_element_list[element_idx],
										 output_streams=[g.SG1_E[output_stream_offset+byte_idx]])
										)


			Psi_transformed_real_state_0_byte_st = g.stack( Psi_transformed_real_state_0_byte_st_list, dim=0)
			Psi_transformed_imag_state_0_byte_st = g.stack( Psi_transformed_imag_state_0_byte_st_list, dim=0)
			Psi_transformed_real_state_1_byte_st = g.stack( Psi_transformed_real_state_1_byte_st_list, dim=0)
			Psi_transformed_imag_state_1_byte_st = g.stack( Psi_transformed_imag_state_1_byte_st_list, dim=0)



			Psi_transformed_real_state_0_byte_st = g.reshape( Psi_transformed_real_state_0_byte_st, [element_num,1,320])
			Psi_transformed_imag_state_0_byte_st = g.reshape( Psi_transformed_imag_state_0_byte_st, [element_num,1,320])
			Psi_transformed_real_state_0_copy_byte_st = g.reshape( Psi_transformed_real_state_1_byte_st, [element_num,1,320])
			Psi_transformed_imag_state_0_copy_byte_st = g.reshape( Psi_transformed_imag_state_1_byte_st, [element_num,1,320])

			Psi_transformed_real_state_0_bytes_list.append( Psi_transformed_real_state_0_byte_st )	
			Psi_transformed_imag_state_0_bytes_list.append( Psi_transformed_imag_state_0_byte_st )			
			Psi_transformed_real_state_0_copy_bytes_list.append( Psi_transformed_real_state_0_copy_byte_st )
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
	


	def write_tranformed_states_2_mem_8L( self, Psi_transformed_real_st, Psi_transformed_imag_st, Psi_transformed_real_copy_st, Psi_transformed_imag_copy_st, target_qubit_L_write_map_st_a, target_qubit_L_write_map_st_b):
	
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


		Psi_transformed_real_copy_mt_320_8       = g.reinterpret(Psi_transformed_real_copy_st, dtype=g.int8)
		Psi_transformed_imag_copy_mt_320_8       = g.reinterpret(Psi_transformed_imag_copy_st, dtype=g.int8)			
		Psi_transformed_real_copy_bytes_mt_list  = g.split(Psi_transformed_real_copy_mt_320_8, dim=-2, num_splits=g.float32.size)
		Psi_transformed_imag_copy_bytes_mt_list  = g.split(Psi_transformed_imag_copy_mt_320_8, dim=-2, num_splits=g.float32.size)		
				
		for byte_idx in range(4):
					
			g.mem_scatter( Psi_transformed_real_bytes_mt_list[byte_idx], Psi_real_bytes_mt_list[byte_idx], target_qubit_L_write_map_st_a)
			g.mem_scatter( Psi_transformed_real_copy_bytes_mt_list[byte_idx], Psi_real_bytes_copy_mt_list[byte_idx], target_qubit_L_write_map_st_b)
			g.mem_scatter( Psi_transformed_imag_bytes_mt_list[byte_idx], Psi_imag_bytes_mt_list[byte_idx], target_qubit_L_write_map_st_a)
			g.mem_scatter( Psi_transformed_imag_copy_bytes_mt_list[byte_idx], Psi_imag_bytes_copy_mt_list[byte_idx], target_qubit_L_write_map_st_b)
					
				

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
				
				read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st = self.select_target_qbit_read_map_west_8L()

				# readout quantum state to be stransformed
				Psi_real_state_0_st, Psi_imag_state_0_st, Psi_real_state_1_st, Psi_imag_state_1_st  = \
							 self.read_out_state_2_transform_8L( read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st)

				
				# readout gate elements	
				element_num = Psi_real_state_0_st.shape[0]*Psi_real_state_0_st.shape[1]
						
				gate_first_column_real_st, gate_first_column_imag_st, gate_second_column_real_st, gate_second_column_imag_st = \
							self.read_out_gate_elements_8( element_num )
                            
				Psi_transformed_real_copy_st, Psi_transformed_imag_copy_st, Psi_transformed_real_st, Psi_transformed_imag_st = \
										self.two_complex_dot_product_8L( Psi_real_state_0_st,
														Psi_imag_state_0_st, 
														Psi_real_state_1_st, 
														Psi_imag_state_1_st, 
														gate_first_column_real_st, 
														gate_first_column_imag_st, 
														gate_second_column_real_st, 
														gate_second_column_imag_st)


					
				
				# export the calculated state into memory and set program output

				target_qubit_L_write_map_st_a, target_qubit_L_write_map_st_b = self.select_target_qbit_write_map_east_8L()

				self.write_tranformed_states_2_mem_8L( Psi_transformed_real_st, 
									Psi_transformed_imag_st, 
									Psi_transformed_real_copy_st, 
									Psi_transformed_imag_copy_st, 
									target_qubit_L_write_map_st_a, 
									target_qubit_L_write_map_st_b )
				

				
			with g.ResourceScope(name=f"increase_gate_counters_scope", is_buffered=True, predecessors=[apply_gate_scope], time=None) as increase_gate_counters_scope :
			
				self.increment_gate_counters()	

				
			self.pgm_pkg.compile_program_context(pgm6)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
