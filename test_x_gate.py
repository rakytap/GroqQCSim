import numpy as np
import groq.api as g
import groq.api.tensor as tensor
from groq.runner import tsp
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


class XGateComponent(g.Component):
    def __init__(self):
        super().__init__()
    def build(self, state_vector, num_qubits, target_qbit, time=0):
        print(state_vector.shape, state_vector.shape[1])
        pairs = [[x,x^(1 << (target_qbit-8))] for x in range(state_vector.shape[1] >> 8) if x < x^(2**(target_qbit-8))]

        with g.ResourceScope(name="copy_scope", is_buffered=True, time=time) as copy:
            temp_tensor = state_vector.read(g.SG8[0], time=0)
            sv_copy = temp_tensor.write(storage_req=tensor.create_storage_request(layout="H1(W), -1, S8(8-15)"))

        innter_dims = state_vector.split_inner_splits()
        innter_dims_copy = sv_copy.split_inner_splits()
        pairs_to_concat0_0 = []
        pairs_to_concat0_1 = []
        pairs_to_concat1_0 = []
        pairs_to_concat1_1 = []
        for i in pairs:
            pairs_to_concat0_0.append(innter_dims[i[0]])
            pairs_to_concat0_1.append(innter_dims[i[1]])
            pairs_to_concat1_0.append(innter_dims_copy[i[0]])
            pairs_to_concat1_1.append(innter_dims_copy[i[1]])

        pairs_to_concat0_0 = g.concat_inner_splits(pairs_to_concat0_0)
        pairs_to_concat0_1 = g.concat_inner_splits(pairs_to_concat0_1)
        pairs_to_concat1_0 = g.concat_inner_splits(pairs_to_concat1_0)
        pairs_to_concat1_1 = g.concat_inner_splits(pairs_to_concat1_1)

        with g.ResourceScope(name="xGate", is_buffered=True, time=None, predecessors = [copy]) as pairing:
            pairs0 = g.stack([pairs_to_concat0_0, pairs_to_concat0_0], dim=0)
            pairs0 = pairs0.read(g.SG8[0], time=0).split(num_splits=2, dim=1)
            pairs1 = g.stack([pairs_to_concat1_1, pairs_to_concat1_1], dim=0)
            pairs1 = pairs1.read(g.SG8[1], time=0).split(num_splits=2, dim=1)

            pairs0[0], pairs0[1], pairs1[0], pairs1[1] = (
                pairs1[0].vxm_identity(alus=[0], output_streams=g.SG4[0]),
                pairs1[1].vxm_identity(alus=[1], output_streams=g.SG4[1]),
                pairs0[0].vxm_identity(alus=[2], output_streams=g.SG4[2]), 
                pairs0[1].vxm_identity(alus=[3], output_streams=g.SG4[3])
            )

            pairs0 = g.concat(pairs0, dim=1)
            pairs1 = g.concat(pairs1, dim=1)

            pairs0_0, pairs0_1 = g.split(pairs0, num_splits=2, dim=0)
            pairs1_0, pairs1_1 = g.split(pairs1, num_splits=2, dim=0)

            pairs0_0.write(storage_req=pairs_to_concat0_0.storage_request, name="pairs0_0")
            pairs0_1.write(storage_req=pairs_to_concat1_1.storage_request, name="pairs0_1")
            pairs1_0.write(storage_req=pairs_to_concat1_0.storage_request, name="pairs1_0")
            pairs1_1.write(storage_req=pairs_to_concat0_1.storage_request, name="pairs1_1")

        return state_vector


num_qubits = 10
pow2qb = 1 << num_qubits

# [
# [0+1j],
# [1+0j],
# ....
# ]
t1_mt = g.input_tensor(shape = (2 , pow2qb), dtype = g.float32, name = "input_1",
                       storage_req=tensor.create_storage_request(layout="H1(W), -1, S8(0-7)"))


x_gate = XGateComponent()
result = x_gate(t1_mt, None , 9 , time = 0)
iop_file = g.compile(base_name="x_gate", result_tensor=result)
print(iop_file)

g.write_visualizer_data("add_tensors")
#!groqview x_gate_tensors/visdata.json


x_gate_program = tsp.create_tsp_runner(iop_file)

# t1_data = np.random.array([[0,1,2,3,4,5]], dtype=np.int8)
random_unitary, _ = np.linalg.qr(np.random.uniform(-1, 1, (pow2qb, pow2qb)).astype(np.float32) \
    + np.random.uniform(-1, 1, (pow2qb, pow2qb)).astype(np.float32)*1j)


result = x_gate_program(input_1=np.ascontiguousarray(random_unitary[0].view(np.float32).reshape(pow2qb, 2).T))
result = result['input_1'][0]+result['input_1'][1]*1j

# print(random_unitary[0].view(np.float32).shape)
print(f"Numpy result ---->\n", random_unitary[0])
print(f"Groq result ---->\n", result)
print(f"Validated -->", np.array_equal(random_unitary[0], result))


#################################################### numpy work ####################################################
# The gate is represented as a matrix:
X = np.array([[0, 1],[1, 0]])

def single_qubit_x_gate(qubit):
    # [[0],[1]]
    return np.dot(X, qubit)


def x_gate(state_vector, qubit_index):
    if 2 ** (len(state_vector) - 1).bit_length()  != len(state_vector):
    # if not np.log2(len(state_vector)).is_integer():
        raise Exception("Illigal State Vector")

    qubits_num = int(np.log2(len(state_vector)))
    if len(state_vector) != 2 ** qubits_num or len(state_vector) < 2:
        raise ValuseError("Invalid state vector length")

    if qubits_num == 1 and qubit_index > 0:
        raise ValuseError("In single qubits you cannot have a qubit index more than 0")

    pairs = [[x,x^(2**qubit_index)] for x in range(len(state_vector)) if x < x^(2**qubit_index)]
    result_array = np.empty(shape=state_vector.shape, dtype=state_vector.dtype)
    for x in pairs:
         result_array[x] = single_qubit_x_gate(state_vector[x])

    return result_array


n = 2
random_unitary, _ = np.linalg.qr(np.random.uniform(-1, 1, (n, n)) + np.random.uniform(-1, 1, (n, n))*1j)

# print(random_unitary @ random_unitary.conj().T, "\n", random_unitary)


# print(np.sum(np.abs(random_unitary), axis=0))
# print(np.sum(np.abs(random_unitary), axis=1))

# print(x_gate(random_unitary[0], 0))
# print("------------",(random_unitary[0]))