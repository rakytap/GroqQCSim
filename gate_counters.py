from generate_static_data import Generate_static_data
from memory_layouts import *
import groq.api as g
import numpy as np


	

class Gate_counters(Generate_static_data):
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
		
		
	

	def increment_gate_counters( self ):

		# 3 west side gate counters
		gate_counter_perm_map_selector_shared = g.shared_memory_tensor(self.gate_counter_perm_map_selector_mt, 
                                                                               name="gate_counter_perm_map_selector_shared" )

		


		gate_counter_distributor_maps_west_shared = g.shared_memory_tensor(
                                                                self.gate_counter_distributor_maps_west_mt,   \
   								name="gate_counter_S_control_qbit_selector_shared") 


		gate_counter_target_qubit_L_distributor_maps_write_shared = g.shared_memory_tensor(
                                                                              self.gate_counter_target_qubit_L_distributor_maps_write_mt,
                                                                              name="gate_counter_target_qubit_L_distributor_maps_write_shared" )

		gate_counters_west_mt = g.stack( [gate_counter_perm_map_selector_shared,
                                                  gate_counter_distributor_maps_west_shared,
                                                  gate_counter_target_qubit_L_distributor_maps_write_shared], 
                                                  dim=0 )

		alu_rq_12 = g.tensor.create_alu_request( alus=[12] )
				
		constant_1_address_offset_st = self.constant_1_address_offset_lower_bits_mt.read(streams=[g.SG1_E[0]])	
		gate_counters_west_st        = gate_counters_west_mt.read(streams=[g.SG1_E[4]])
				
		# offset the memory address by 1
		gate_counters_west_st        = g.add(constant_1_address_offset_st, 
                                                          gate_counters_west_st, 
                                                          alus=alu_rq_12, 
                                                          output_streams=g.SG4_W[1], 
                                                          overflow_mode=g.OverflowMode.MODULAR)
				
		# save the offsetted memory address
		gate_counters_west_st.write(name=f"gate_counters_west", \
						storage_req=gate_counters_west_mt.storage_request, 
						time=0)#, program_output=True)



		
		# 6 east side gate counters

		gate_counter_gate_element_selector_00_shared = g.shared_memory_tensor( \
							self.gate_counter_gate_element_selector_00_mt, \
							name="gate_counter_gate_element_selector_00_shared") 
				
		gate_counter_gate_element_selector_01_shared = g.shared_memory_tensor( \
							self.gate_counter_gate_element_selector_01_mt, \
							name="gate_counter_gate_element_selector_01_shared") 

				
		gate_counter_gate_element_selector_10_shared = g.shared_memory_tensor( \
							self.gate_counter_gate_element_selector_10_mt, \
							name="gate_counter_gate_element_selector_10_shared") 
				
		gate_counter_gate_element_selector_11_shared = g.shared_memory_tensor( \
							self.gate_counter_gate_element_selector_11_mt, \
							name="gate_counter_gate_element_selector_11_shared") 


		gate_counter_S_control_qbit_selector_shared = g.shared_memory_tensor(
                                                                 self.gate_counter_S_control_qbit_selector_mt,   \
   								 name="gate_counter_S_control_qbit_selector_shared") 


		gate_counter_gate_filtering_selector_shared = g.shared_memory_tensor(
                                                                  self.gate_counter_gate_filtering_selector_mt,   \
                                                                  name="gate_counter_gate_filtering_selector_shared") 


		gate_counters_east_mt = g.stack( [gate_counter_gate_filtering_selector_shared,
                                                  gate_counter_S_control_qbit_selector_shared,
                                                  gate_counter_gate_element_selector_00_shared,
                                                  gate_counter_gate_element_selector_01_shared,
                                                  gate_counter_gate_element_selector_10_shared,
                                                  gate_counter_gate_element_selector_11_shared],       
                                                  dim=0 )




		alu_rq_15 = g.tensor.create_alu_request( alus=[15] )			
			
				
		constant_1_address_offset_st     = self.constant_1_address_offset_lower_bits_mt.read(streams=[g.SG1_E[0]])				

		gate_counters_east_st  = gate_counters_east_mt.read(streams=[g.SG1_W[4]])
				
		# offset the memory address by 1
		gate_counters_east_st  = g.add(constant_1_address_offset_st, 
                                                          gate_counters_east_st, 
                                                          alus=alu_rq_15, 
                                                          output_streams=g.SG4_E[6],
                                                          overflow_mode=g.OverflowMode.MODULAR)
				
		# save the offsetted memory address
		gate_counters_east_st.write( name=f"gate_counters_east", 
					     storage_req=gate_counters_east_mt.storage_request, 
					     time=1)#, program_output=True)
		
			

		

