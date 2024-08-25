from split_gates import Split_gates
from memory_layouts import *
import groq.api as g
import numpy as np


	

class Gate_S_base(Split_gates):
	"""
    A base class used to implement the gate operation on qubits S.
    S qubits are qubits 0,1,2,...,7


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




	def prepare_gate_data( self ):
		"""
		Function to generate memory tensors to be used in the calculations.
        """

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

		layout_real_diag     = layout_gate_kernels_real + f", A1({address_offset})"
		layout_imag_diag     = layout_gate_kernels_imag + f", A1({address_offset})"		
		layout_real_off_diag = layout_gate_kernels_real_copy + f", A1({address_offset})"
		layout_imag_off_diag = layout_gate_kernels_imag_copy + f", A1({address_offset})"	
				
		storage_req_real_diag     = g.tensor.create_storage_request( layout = layout_real_diag )
		storage_req_imag_diag     = g.tensor.create_storage_request( layout = layout_imag_diag )
		storage_req_real_off_diag = g.tensor.create_storage_request( layout = layout_real_off_diag )
		storage_req_imag_off_diag = g.tensor.create_storage_request( layout = layout_imag_off_diag )


		return State_input_real_shared, State_input_imag_shared, State_input_real_copy_shared, State_input_imag_copy_shared, \
                       state_1_broadcasted_shared, state_0_broadcasted_shared, \
                       gate_kernels_broadcasted_real_shared_dict, gate_kernels_broadcasted_imag_shared_dict, gate_kernels_broadcasted_real_shared_copy_dict, gate_kernels_broadcasted_imag_shared_copy_dict, \
                       storage_req_real_diag, storage_req_imag_diag, storage_req_real_off_diag, storage_req_imag_off_diag 


	def gather_permutation_matrix( self, mxm_permute_map_selector, mxm_permute_matrices):
		"""
		Function used to gather the permutation matrices for MXM permute via memory map uploaded from CPU for a specific target qubit S.
		 
       
        Parameters
        ----------
        mxm_permute_map_selector : MemoryTensor
            The memory map to gather the permutation matrix
            
        mxm_permute_matrices : MemoryTensor
            The memory tensor containing the permutation matrices

        Return
        ------
        Returns with the stream tensor of the permutation matrix.
        """				

		### split the permutation matrices over slices ###
							
		# split the stored permutation matrices into 16-line chunks
		# one permute matrix will occupy 20 addresses
		mxm_permute_rows           = g.reshape( mxm_permute_matrices, [ small_qbit_num_limit*320, 1, 320 ] )
		mxm_permute_rows_list      = g.split( mxm_permute_rows, dim=0, num_splits=small_qbit_num_limit*320 )
		mxm_permute_mt_slices_list = []
					
		for slice_idx in range(16):
			mxm_permute_mt_slice = mxm_permute_rows_list[slice_idx:small_qbit_num_limit*320:16]
			mxm_permute_mt_slices_list.append( g.concat(mxm_permute_mt_slice) )

		# the permutation matrix occupy 20 adresses over 16 slices
		mxm_permute_row_selectors_list = g.split( mxm_permute_map_selector, dim=0, num_splits = 20 )

		mxm_permute_row_st_list = []
		# the permutation matrix occupy 20 adresses over 16 slices
		for permute_selector_idx in range(20):
                                       
			if mxm_permute_row_selectors_list[permute_selector_idx].is_memory_tensor :                          
				mxm_permute_row_selector_st = mxm_permute_row_selectors_list[permute_selector_idx].read( streams = g.SG1[31])
			else:
				mxm_permute_row_selector_st = mxm_permute_row_selectors_list[permute_selector_idx]  # already given by a stream tensor
						                
			for stream_idx in range(16):

				mxm_permute_row_st = g.mem_gather(mxm_permute_mt_slices_list[stream_idx], mxm_permute_row_selector_st, output_streams=[g.SG1[stream_idx]] )
				mxm_permute_row_st_list.append( mxm_permute_row_st )
                                        	
                                        	
                                        	
		permute_matrix_st = g.concat( mxm_permute_row_st_list )
	
		return permute_matrix_st



	def permutation_of_S_qubits(self, Psi_transformed_real_mt, Psi_transformed_imag_mt, Psi_transformed_real_copy_mt, Psi_transformed_imag_copy_mt):
		"""
		Function used to permute the qubits S to generate the state vector pair for the linear transform.
		Permutation is done by the two permutors on the chip: the real part on the east side, the imaginary part on the west side. 
		 
       
        Parameters
        ----------
        Psi_transformed_real_mt : MemoryTensor
            The memory tensor of the real part of the state (float32)
            
        Psi_transformed_imag_mt : MemoryTensor
            The memory tensor of the imaginary part of the state (float32)
            
        Psi_transformed_real_copy_mt : MemoryTensor
            The memory tensor of the real part of the copy of the state (float32)
            
        Psi_transformed_imag_copy_mt : MemoryTensor
            The memory tensor of the imaginary part of the copy of the state (float32)            
  

        Return
        ------
        Returns with the memory tensors of the permuted state (real, imag)
        """			
        
        
		# create reusable permutor requests
		permutor_requests = []
		permutor_requests.append( g.tensor.create_permutor_request( [0], 1 ) )
		permutor_requests.append( g.tensor.create_permutor_request( [1], 1 ) )

		state_permute_map_selector_shared = g.shared_memory_tensor(self.state_permute_map_selector_mt, name="state_permute_map_selector_shared")		
		state_permute_map_selector_list   = g.split_vectors(input=state_permute_map_selector_shared, splits=[1]*gate_count)
			
		# otherwise the permutor slices are used
		permute_maps_mt_west_shared = g.shared_memory_tensor(self.permute_maps_mt_west, name="permute_maps_west_shared")
		permute_maps_mt_east_shared = g.shared_memory_tensor(self.permute_maps_mt_east, name="permute_maps_east_shared")

		mxm_permute_matrices_west_shared = None
		mxm_permute_matrices_east_shared = None

		state_permute_map_selector_west_st = state_permute_map_selector_list[0].read(streams=[g.SG1_W[24]])
		state_permute_map_selector_east_st = state_permute_map_selector_list[0].read(streams=[g.SG1_E[24]])
		permute_map_west_st                = g.mem_gather(permute_maps_mt_west_shared, state_permute_map_selector_west_st, output_streams=[g.SG1_W[24]])
		permute_map_east_st                = g.mem_gather(permute_maps_mt_east_shared, state_permute_map_selector_east_st, output_streams=[g.SG1_E[24]])

		state_real_mt_8 = g.reinterpret(Psi_transformed_real_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
		state_imag_mt_8 = g.reinterpret(Psi_transformed_imag_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input

		state_real_st_8 = state_real_mt_8.read( streams=g.SG1_E[4] )
		state_imag_st_8 = state_imag_mt_8.read( streams=g.SG1_W[4] )

					
		permuted_state_real_st_8 = g.permute_inner(state_real_st_8, permute_map_west_st, permutor_req=permutor_requests[1], input_streams=[g.SG1_E[0], g.SG1_E[24]], output_streams=g.SG1[12], time=0 )
		permuted_state_imag_st_8 = g.permute_inner(state_imag_st_8, permute_map_east_st, permutor_req=permutor_requests[0], input_streams=[g.SG1_W[0], g.SG1_W[24]], output_streams=g.SG1[4], time=0 )

		memory_addresses = permuted_state_real_st_8.physical_shape.outer_size >> 2

		permuted_state_real_st_8 = g.reshape(permuted_state_real_st_8, [memory_addresses, 4, matrix_size] )
		permuted_state_imag_st_8 = g.reshape(permuted_state_imag_st_8, [memory_addresses, 4, matrix_size] )
				
		permuted_state_real_st = g.reinterpret(permuted_state_real_st_8, g.float32 )
		permuted_state_imag_st = g.reinterpret(permuted_state_imag_st_8, g.float32 )

		permuted_state_real_mt = permuted_state_real_st.write(name=f"permuted_result_real", storage_req=Psi_transformed_real_copy_mt.storage_request)
		permuted_state_imag_mt = permuted_state_imag_st.write(name=f"permuted_result_imag", storage_req=Psi_transformed_imag_copy_mt.storage_request)


		return permuted_state_real_mt, permuted_state_imag_mt


	def MXM_permutation_of_S_qubits(self, Psi_transformed_real_mt, Psi_transformed_imag_mt, Psi_transformed_real_copy_mt, Psi_transformed_imag_copy_mt):
		"""
		Function used to permute the qubits S to generate the state vector pair for the linear transform.
		Permutation is done by the four MXM planes on the chip. 
		 
       
        Parameters
        ----------
        Psi_transformed_real_mt : MemoryTensor
            The memory tensor of the real part of the state (float32)
            
        Psi_transformed_imag_mt : MemoryTensor
            The memory tensor of the imaginary part of the state (float32)
            
        Psi_transformed_real_copy_mt : MemoryTensor
            The memory tensor of the real part of the copy of the state (float32)
            
        Psi_transformed_imag_copy_mt : MemoryTensor
            The memory tensor of the imaginary part of the copy of the state (float32)            
  

        Return
        ------
		Returns with the memory tensors of the permuted state (real, imag)
        """			
        
        
		mxm_requests = []
		mxm_requests.append(g.tensor.create_mxm_request(planes=[0], signed=True))
		mxm_requests.append(g.tensor.create_mxm_request(planes=[1], signed=True))
		mxm_requests.append(g.tensor.create_mxm_request(planes=[2], signed=True))
		mxm_requests.append(g.tensor.create_mxm_request(planes=[3], signed=True))


		mxm_permute_matrices_west_shared = g.shared_memory_tensor(self.mxm_permute_matrices_mt_west, name="mxm_permute_matrices_west_shared")
		mxm_permute_matrices_east_shared = g.shared_memory_tensor(self.mxm_permute_matrices_mt_east, name="mxm_permute_matrices_east_shared")
				
		mxm_permute_map_selector_shared  = g.shared_memory_tensor(self.state_permute_map_selector_expanded, name="mxm_permute_map_selector_shared")

		mxm_permute_matrices_west_list = g.split(mxm_permute_matrices_west_shared, dim=0, num_splits=8 )
		mxm_permute_matrices_east_list = g.split(mxm_permute_matrices_east_shared, dim=0, num_splits=8 )
					
		# gathering the permutation matrices over the 16 slices with mem_gather 
		# allows dynamic memory addressing according to the target qubit
		permute_matrix_west_1_st = self.gather_permutation_matrix( mxm_permute_map_selector_shared, mxm_permute_matrices_west_shared)									
		permute_matrix_west_2_st = self.gather_permutation_matrix( mxm_permute_map_selector_shared, mxm_permute_matrices_west_shared)
		permute_matrix_east_1_st = self.gather_permutation_matrix( mxm_permute_map_selector_shared, mxm_permute_matrices_east_shared)	
		permute_matrix_east_2_st = self.gather_permutation_matrix( mxm_permute_map_selector_shared, mxm_permute_matrices_east_shared)

					
		# list storing the fetched rows of the permutation matrix
		mxm_permute_row_west_st_list = []					
			
		# mxm only accepts tensors with an inner dim of 320, thus 256 inner splits should be extended to 320
		Psi_transformed_real_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=Psi_transformed_real_mt.storage_request, initialized=False)
		Psi_transformed_imag_mt_320 = g.constant_tensor((memory_slices, 320), dtype=g.float32, storage_req=Psi_transformed_imag_mt.storage_request, initialized=False)

		Psi_transformed_real_mt_320 = g.reinterpret(Psi_transformed_real_mt_320, dtype=g.int8)
		Psi_transformed_imag_mt_320 = g.reinterpret(Psi_transformed_imag_mt_320, dtype=g.int8)

		Psi_transformed_real_copy_mt = g.reinterpret(Psi_transformed_real_copy_mt, dtype=g.int8)
		Psi_transformed_imag_copy_mt = g.reinterpret(Psi_transformed_imag_copy_mt, dtype=g.int8)

		# separating the tensors along the slice dimension (separating the bytes)
		Psi_transformed_real_mt_list      = g.split(Psi_transformed_real_mt_320, dim=-2, num_splits=g.float32.size)
		Psi_transformed_imag_mt_list      = g.split(Psi_transformed_imag_mt_320, dim=-2, num_splits=g.float32.size)
		Psi_transformed_real_copy_mt_list = g.split(Psi_transformed_real_copy_mt, dim=-2, num_splits=g.float32.size)
		Psi_transformed_imag_copy_mt_list = g.split(Psi_transformed_imag_copy_mt, dim=-2, num_splits=g.float32.size)
						
					
		# 20 cycles to read the permute matrix, 8 for installing the weights and each matmul operation takes "memory_slices" cycles
		time_list = [8, 28, 8+memory_slices, 28+memory_slices]
		weights_list = []
		weights_list.append(g.install_weights(permute_matrix_west_1_st, planes=mxm_requests[0], time=0))
		weights_list.append(g.install_weights(permute_matrix_west_2_st, planes=mxm_requests[1], time=20))					
		weights_list.append(g.install_weights(permute_matrix_east_1_st, planes=mxm_requests[2], time=-26)) #TODO make it time=0
		weights_list.append(g.install_weights(permute_matrix_east_2_st, planes=mxm_requests[3], time=20-26)) #TODO make it time=20
					
		for byte_idx in range(4):
					
			# Real part
			permuted_state_real_st = Psi_transformed_real_mt_list[byte_idx].matmul(weights_list[byte_idx%2+2], planes=mxm_requests[byte_idx%2+2], time=time_list[byte_idx])
			permuted_state_real_st = g.reinterpret(permuted_state_real_st, dtype=g.int8)
			# we only need the last byte from the MXM result
			permuted_state_real_st = g.split(permuted_state_real_st, dim=-2, num_splits=g.int32.size)[0]
			permuted_state_real_st.write(storage_req=Psi_transformed_real_copy_mt_list[byte_idx].storage_request)

			# Imag part
			permuted_state_imag_st = Psi_transformed_imag_mt_list[byte_idx].matmul(weights_list[byte_idx%2], planes=mxm_requests[byte_idx%2], time=time_list[byte_idx])
			permuted_state_imag_st = g.reinterpret(permuted_state_imag_st, dtype=g.int8)
			# we only need the last byte from the MXM result
			permuted_state_imag_st = g.split(permuted_state_imag_st, dim=-2, num_splits=g.int32.size)[0]
			permuted_state_imag_st.write(storage_req=Psi_transformed_imag_copy_mt_list[byte_idx].storage_request)

		Psi_transformed_real_copy_mt = g.reinterpret(Psi_transformed_real_copy_mt, dtype=g.float32)
		Psi_transformed_imag_copy_mt = g.reinterpret(Psi_transformed_imag_copy_mt, dtype=g.float32)

		permuted_state_real_mt = g.constant_tensor(Psi_transformed_real_copy_mt.shape, g.float32, name="permuted_result_real", storage_req=self.state_real_copy_request, initialized=False)
		permuted_state_imag_mt = g.constant_tensor(Psi_transformed_imag_copy_mt.shape, g.float32, name="permuted_result_imag", storage_req=self.state_imag_copy_request, initialized=False)

		return permuted_state_real_mt, permuted_state_imag_mt










	def VXM_S_round_trip_st(self, state_real_st, state_imag_st, state_permuted_real_st, state_permuted_imag_st, gate_diag_real_mt, gate_diag_imag_mt, gate_off_diag_real_mt, gate_off_diag_imag_mt):

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
		#state_real_st 		       = states_real_mt.read( streams=[g.SG4[0]] )	
		gate_diag_real_mt_256         = gate_diag_real_mt#g.constant_tensor((1,256,), g.float32, storage_req=storage_req_real_diag, name="gate_diag_real_mt_256", initialized=False)	
		gate_diag_real_st 	       = gate_diag_real_mt_256.read( streams=[g.SG4_W[0]] )
		state_real__gate_diag_real_st = g.mul( state_real_st, gate_diag_real_st, alus=alu_rq_12, output_streams=g.SG4_E[0] )#, time=150 )
				

		# calculate Psi_imag * gate_diag_imag
		#state_imag_st 		       = states_imag_mt.read( streams=[g.SG4[3]] )
		gate_diag_imag_mt_256         = gate_diag_imag_mt#g.constant_tensor((1,256,), g.float32, storage_req=storage_req_imag_diag, name="gate_diag_imag_mt_256", initialized=False)	
		gate_diag_imag_st 	       = gate_diag_imag_mt_256.read( streams=[g.SG4_W[3]] )
		state_imag__gate_diag_imag_st = g.mul( state_imag_st, gate_diag_imag_st, alus=[4], output_streams=g.SG4_E[2] )	
				

		# calculate Psi_real * gate_diag_real - Psi_imag * gate_diag_imag
		state__gate_diag_real_st	= g.sub( state_real__gate_diag_real_st, state_imag__gate_diag_imag_st, alus=[2], output_streams=g.SG4_E[2] )
						
				

	
					
					
		# calculate Psi_imag * gate_diag_real
		state_imag__gate_diag_real_st	= g.mul( state_imag_st, gate_diag_real_st, alus=alu_rq_1, output_streams=g.SG4_E[3] )		
						
		# calculate Psi_real * gate_diag_imag
		state_real__gate_diag_imag_st	= g.mul( state_real_st, gate_diag_imag_st, alus=[0], output_streams=g.SG4_E[1] )
				

		# calculate Psi_real * gate_diag_imag + Psi_imag * gate_diag_real
		state__gate_diag_imag_st	= g.add( state_imag__gate_diag_real_st, state_real__gate_diag_imag_st, alus=[3], output_streams=g.SG4_W[1]  )
				
					
						
					
		# calculate Psi_permuted_real * gate_off_diag_real
		#state_permuted_real_st     = permuted_state_real_mt.read( streams=[g.SG4_E[6]] )
		gate_off_diag_real_mt_256 = gate_off_diag_real_mt#g.constant_tensor((1,256,), g.float32, storage_req=storage_req_real_off_diag, name="gate_off_diag_real_mt_256", initialized=False)
		gate_off_diag_real_st 	                    = gate_off_diag_real_mt_256.read( streams=[g.SG4_W[6]] )
		state_permuted_real__gate_off_diag_real_st = g.mul( state_permuted_real_st, gate_off_diag_real_st, alus=[13], output_streams=g.SG4_E[6] )

		# calculate Psi_permuted_imag * gate_off_diag_imag
		#state_permuted_imag_st 	   = permuted_state_imag_mt.read( streams=[g.SG4[4]] )
		gate_off_diag_imag_mt_256 = gate_off_diag_imag_mt#g.constant_tensor((1,256,), g.float32, storage_req=storage_req_imag_off_diag, name="gate_off_diag_imag_mt_256", initialized=False)
		gate_off_diag_imag_st                      = gate_off_diag_imag_mt_256.read( streams=[g.SG4_W[5]] )
		state_permuted_imag__gate_off_diag_imag_st = g.mul( state_permuted_imag_st, gate_off_diag_imag_st, alus=[5], output_streams=g.SG4_E[5] )	

		# calculate Psi_permuted_real * gate_off_diag_real - Psi_permuted_imag * gate_off_diag_imag
		state_permuted__gate_off_diag_real_st	= g.sub( state_permuted_real__gate_off_diag_real_st, state_permuted_imag__gate_off_diag_imag_st, alus=[10], output_streams=g.SG4_E[5]  )	

					
				
		# calculate Psi_permuted_imag * gate_off_diag_real
		state_permuted_imag__gate_off_diag_real_st	= g.mul( state_permuted_imag_st, gate_off_diag_real_st, alus=[9], output_streams=g.SG4_E[4] )
					
		# calculate Psi_permuted_real * gate_diag_imag
		state_permuted_real__gate_off_diag_imag_st	= g.mul( state_permuted_real_st, gate_off_diag_imag_st, alus=[8], output_streams=g.SG4_E[7] )
				
		# calculate Psi_permuted_real * gate_off_diag_imag + Psi_permuted_imag * gate_off_diag_real
		state_permuted__gate_off_diag_imag_st	= g.add( state_permuted_imag__gate_off_diag_real_st, state_permuted_real__gate_off_diag_imag_st, alus=[11], output_streams=g.SG4_W[7]  )
				
		

		# calculate (Psi*gate_diag).real + (Psi_permuted*gate_off_diag).real
		Psi_transformed_real_st	= g.add( state__gate_diag_real_st,  state_permuted__gate_off_diag_real_st, alus=[7], output_streams=g.SG4_W[2] )
		Psi_transformed_real_st = g.vxm_identity( Psi_transformed_real_st, alus=[6], output_streams=g.SG4_W[2] )

		# calculate (Psi*gate_diag).imag + (Psi_permuted*gate_off_diag).imag
		Psi_transformed_imag_st	    = g.add( state__gate_diag_imag_st,  state_permuted__gate_off_diag_imag_st, alus=[14], output_streams=g.SG4_E[6] )
		Psi_transformed_imag_st = g.vxm_identity( Psi_transformed_imag_st, alus=[15], output_streams=g.SG4_W[0] )


		return Psi_transformed_real_st, Psi_transformed_imag_st



	def VXM_S_round_trip(self, state_real_mt, state_imag_mt, permuted_state_real_mt, permuted_state_imag_mt, gate_diag_real_mt, gate_diag_imag_mt, gate_off_diag_real_mt, gate_off_diag_imag_mt):

		# the number of inner split over which the state is distributed
		inner_splits = state_real_mt.shape[0]

		# define state with 320 vectors instead of 256 inner splits
		state_real_mt_320      = g.constant_tensor((inner_splits, 320), dtype=g.float32, storage_req=state_real_mt.storage_request, initialized=False)		
		state_imag_mt_320      = g.constant_tensor((inner_splits, 320), dtype=g.float32, storage_req=state_imag_mt.storage_request, initialized=False)
		state_real_copy_mt_320 = g.constant_tensor((inner_splits, 320), dtype=g.float32, storage_req=permuted_state_real_mt.storage_request, initialized=False)
		state_imag_copy_mt_320 = g.constant_tensor((inner_splits, 320), dtype=g.float32, storage_req=permuted_state_imag_mt.storage_request, initialized=False)		


		state_real_st 		       = state_real_mt_320.read( streams=[g.SG4[0]], time=0 )	
		state_imag_st 		       = state_imag_mt_320.read( streams=[g.SG4[3]] )
		state_permuted_real_st         = state_real_copy_mt_320.read( streams=[g.SG4_E[6]] )
		state_permuted_imag_st 	       = state_imag_copy_mt_320.read( streams=[g.SG4[4]] )

		Psi_transformed_real_st, Psi_transformed_imag_st = self.VXM_S_round_trip_st(state_real_st, 
                                                                                            state_imag_st, 
                                                                                            state_permuted_real_st, 
                                                                                            state_permuted_imag_st, 
                                                                                            gate_diag_real_mt, 
                                                                                            gate_diag_imag_mt, 
                                                                                            gate_off_diag_real_mt, 
                                                                                            gate_off_diag_imag_mt)

		state_real_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real", storage_req=state_real_mt.storage_request)
		Psi_transformed_real_copy_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real_copy", storage_req=permuted_state_real_mt.storage_request)


		state_imag_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag", storage_req=state_imag_mt.storage_request)
		Psi_transformed_imag_copy_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag_copy", storage_req=permuted_state_imag_mt.storage_request)
		
		return state_real_mt, state_imag_mt
		
