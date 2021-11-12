import numpy as np


class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """
        probabilities = []
        
        # Convert strings to indices
        e_dict = dict(zip(self.emissions, list(range(self.M))))    
        e_sequence = []
        for item in seq:
            e_sequence.append(e_dict[item])

        # First Maximal Probability
        probabilities.append(
            tuple(
                self.pi[state]*self.B[state, e_sequence[0]]
                for state in range(self.N))
        )

        for i in range(1, len(e_sequence)):
            previous_probabilities = probabilities[-1]
            current_probabilities = []
            for col in range(len(self.A[0,:])):
                p = max(
                    previous_probabilities[state]*self.A[state,col]*self.B[col,e_sequence[i]]
                    for state in range(self.N)
                )
                current_probabilities.append(p)
            probabilities.append(tuple(current_probabilities))

        # Find the sequence of hidden states
        hidden_states_sequence = []
        for i in probabilities:
            hidden_state = self.states[np.argmax(i)]
            hidden_states_sequence.append(hidden_state)
            
        return hidden_states_sequence