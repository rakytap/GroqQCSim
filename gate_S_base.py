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
