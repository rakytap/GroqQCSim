from split_gates import Split_gates
from memory_layouts import *
import groq.api as g
import numpy as np


	

class Gate_S_base(Split_gates):


	def __init__(self, pgm_pkg, mxm_permutation=True):
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation )



	def gather_permutation_matrix( self, mxm_permute_map_selector, mxm_permute_matrices):

		### split the permutation matrices over slices ###
							
		# split the stored permutation matrices into 16-line chunks
		# one permute matrix will occupy 20 addresses
		mxm_permute_rows = g.reshape( mxm_permute_matrices, [ small_qbit_num_limit*320, 1, 320 ] )
		mxm_permute_rows_list = g.split( mxm_permute_rows, dim=0, num_splits=small_qbit_num_limit*320 )
		mxm_permute_mt_slices_list = []
					
		for slice_idx in range(16):
			mxm_permute_mt_slice = mxm_permute_rows_list[slice_idx:small_qbit_num_limit*320:16]
			mxm_permute_mt_slices_list.append( g.concat(mxm_permute_mt_slice) )

		# the permutation matrix occupy 20 adresses over 16 slices
		mxm_permute_row_selectors_list = g.split( mxm_permute_map_selector, dim=0, num_splits = 20 )

		mxm_permute_row_st_list = []
		# the permutation matrix occupy 20 adresses over 16 slices
		for permute_selector_idx in range(20):
                                       
			mxm_permute_row_selector_st = mxm_permute_row_selectors_list[permute_selector_idx].read( streams = g.SG1[31])
						                
			for stream_idx in range(16):

				mxm_permute_row_st = g.mem_gather(mxm_permute_mt_slices_list[stream_idx], mxm_permute_row_selector_st, output_streams=[g.SG1[stream_idx]] )
				mxm_permute_row_st_list.append( mxm_permute_row_st )
                                        	
                                        	
                                        	
		permute_matrix_st = g.concat( mxm_permute_row_st_list )
	
		return permute_matrix_st





	def VXM_S_round_trip(self, Psi_transformed_real_mt, Psi_transformed_imag_mt, permuted_state_real_mt, permuted_state_imag_mt, gate_diag_real_mt, gate_diag_imag_mt, gate_off_diag_real_mt, gate_off_diag_imag_mt):

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
		state_real_st 		       = Psi_transformed_real_mt.read( streams=[g.SG4[0]] )	
		#gate_diag_real_mt_256         = g.constant_tensor((1,256,), g.float32, storage_req=storage_req_real_diag, name="gate_diag_real_mt_256", initialized=False)	
		#gate_diag_real_st 	       = gate_diag_real_mt_256.read( streams=[g.SG4_W[0]] )
		#state_real__gate_diag_real_st = g.mul( state_real_st, gate_diag_real_st, alus=alu_rq_12, output_streams=g.SG4_E[0], time=150 )
		gate_diag_real_mt.latch(alus=alu_rq_12, input_streams=g.SG4_W[0], time=0)
		state_real__gate_diag_real_st = state_real_st.mul(None, alus=alu_rq_12, output_streams=g.SG4_E[0], time=2)
				

		# calculate Psi_imag * gate_diag_imag
		state_imag_st 		       = Psi_transformed_imag_mt.read( streams=[g.SG4[3]] )
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

		Psi_transformed_real_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real", storage_req=self.state_real_request)
		Psi_transformed_real_copy_mt = Psi_transformed_real_st.write(name=f"Psi_transformed_real_copy", storage_req=self.state_real_copy_request)


		# calculate (Psi*gate_diag).imag + (Psi_permuted*gate_off_diag).imag
		Psi_transformed_imag_st	    = g.add( state__gate_diag_imag_st,  state_permuted__gate_off_diag_imag_st, alus=[14], output_streams=g.SG4_E[6] )
		Psi_transformed_imag_st = g.vxm_identity( Psi_transformed_imag_st, alus=[15], output_streams=g.SG4_W[0] )


		Psi_transformed_imag_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag", storage_req=self.state_imag_request)
		Psi_transformed_imag_copy_mt = Psi_transformed_imag_st.write(name=f"Psi_transformed_imag_copy", storage_req=self.state_imag_copy_request)

		return Psi_transformed_imag_mt, Psi_transformed_imag_mt
					
	
