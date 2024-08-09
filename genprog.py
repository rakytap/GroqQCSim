from deap import base, creator, tools, gp, algorithms
import numpy as np
from scipy.io import loadmat
import functools
from squander import Circuit, utils

ANGLE = np.float64
QUBIT = np.int32

class CircuitBuilder(Circuit):
    def __init__(self, num_qubits):
        super().__init__(num_qubits)
        self.params = np.empty(0, dtype=ANGLE)
    def add_U3_params(self, theta, phi, lambd, qubit):
        self.add_U3(qubit)
        self.params = np.concatenate([self.params, np.array([theta, phi, lambd])])
        return self
    def add_CRY_params(self, theta, control_qubit, target_qubit):
        self.add_CRY(target_qubit, control_qubit)
        self.params = np.concatenate([self.params, np.array([theta])])
        return self
    def add_CNOT_params(self, control_qubit, target_qubit):
        self.add_CNOT(target_qubit, control_qubit)
        return self
    def add_circuit_params(self, circuit):
        self.add_Circuit(circuit)
        self.params = np.concatenate([self.params, circuit.params])
        return self

def U3_gate(num_qubits, theta, phi, lambd, qubit):
    return CircuitBuilder(num_qubits).add_U3_params(theta, phi, lambd, qubit)
    
def CRY_gate(num_qubits, theta, control_qubit, target_qubit):
    return CircuitBuilder(num_qubits).add_CRY_params(theta, control_qubit, target_qubit)

def CNOT_gate(num_qubits, control_qubit, target_qubit):
    return CircuitBuilder(num_qubits).add_CNOT_params(control_qubit, target_qubit)

def sequence(num_qubits, circuit1, circuit2):
    return CircuitBuilder(num_qubits).add_circuit_params(circuit1).add_circuit_params(circuit2)

def pass_through(x): return x
    
def evaluate(individual, pset, hof, unitary, num_qubits, cost_func_method):
    compiled_circuit = gp.compile(expr=individual, pset=pset)
    #print(individual)
    circuit = CircuitBuilder(num_qubits)
    built = compiled_circuit(circuit)
    result = unitary.copy()
    built.apply_to( built.params, result)
    return (np.abs(np.trace(np.real(result)) - len(result)),)
def decompose_unitary(filepath, layers=4):
    ## load the unitary from file
    #data = loadmat(filepath)
    ## The unitary to be decomposed  
    #Umtx = data['Umtx']
    #num_qubits = len(Umtx).bit_length()
    
    from qiskit import QuantumCircuit, transpile
    from squander.utils import get_unitary_from_qiskit_circuit
    qc_trial = QuantumCircuit.from_qasm_file( filepath )
    qc_trial = transpile(qc_trial, optimization_level=3, basis_gates=['cz', 'cx', 'u3'], layout_method='sabre')
    num_qubits = qc_trial.num_qubits 
    Umtx = get_unitary_from_qiskit_circuit( qc_trial )
    
    pset = gp.PrimitiveSetTyped("MAIN", [CircuitBuilder], CircuitBuilder)
    
    pset.addPrimitive(functools.partial(U3_gate, num_qubits), [ANGLE, ANGLE, ANGLE, QUBIT], CircuitBuilder, name="U3")
    pset.addPrimitive(functools.partial(CNOT_gate, num_qubits), [QUBIT, QUBIT], CircuitBuilder, name="CNOT")
    #pset.addPrimitive(functools.partial(CRY_gate, num_qubits), [ANGLE, QUBIT, QUBIT], CircuitBuilder, name="CRY")
    
    pset.addPrimitive(functools.partial(sequence, num_qubits), [CircuitBuilder, CircuitBuilder], CircuitBuilder, name="sequence")
    
    pset.addPrimitive(pass_through, [ANGLE], ANGLE, name="anglepass")
    pset.addPrimitive(pass_through, [QUBIT], QUBIT, name="qubitpass")
    
    def randfloat_in_range(lo, hi): return ((hi-lo)*np.random.rand(1)-lo)[0]
    pset.addEphemeralConstant("angle_param", functools.partial(randfloat_in_range, 0, 2*np.pi), ret_type=ANGLE)
    pset.addEphemeralConstant("qubit_index", functools.partial(np.random.randint, 0, num_qubits), ret_type=QUBIT)

    import operator
    # Set up the DEAP framework
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    tree_depth = num_qubits * (num_qubits - 1) * layers
    # Register the genetic operators
    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    hof = tools.HallOfFame(1)
    #toolbox/pset contains wrapper functions
    cost_func_method = 0
    toolbox.register("evaluate", functools.partial(evaluate, pset=pset, hof=hof, unitary=Umtx, num_qubits=num_qubits, cost_func_method=cost_func_method))
    toolbox.register("select", tools.selTournament, tournsize=5)
    #toolbox.register("select", tools.selBest)
    #toolbox.register("select", tools.selDoubleTournament, fitness_size=10, parsimony_size=1.4, fitness_first=False)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, pset=pset, min_=0, max_=2)
    npopulation = 32*32
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=tree_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=tree_depth))
    toolbox.register("generate", toolbox.population, n=npopulation)
    pop = toolbox.generate()
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=100000, stats=stats, halloffame=hof, verbose=True)
    #algorithms.eaMuPlusLambda(pop, toolbox, mu=10, lambda_=100, cxpb=0.6, mutpb=0.3, ngen=100000, stats=stats, halloffame=hof, verbose=True)
    #algorithms.eaMuCommaLambda(pop, toolbox, mu=10, lambda_=100, cxpb=0.6, mutpb=0.3, ngen=100000, stats=stats, halloffame=hof, verbose=True)
    #algorithms.eaGenerateUpdate(toolbox, ngen=100000, stats=stats, halloffame=hof, verbose=True)
    print("Best individual:", hof[0])
    print("Best individual fitness:", hof[0].fitness.values)
    
decompose_unitary("/home/morse/squander/sequential-quantum-gate-decomposer/benchmarks/IBM/alu-v4_37.qasm")
