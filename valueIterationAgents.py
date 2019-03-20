# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        indicated_iterations = iterations
        arg_max = float("-inf")
        state_list = mdp.getStates()
        #label = util.Counter()
        iter_score = 0
        i = 0
        while i < indicated_iterations:
            label = util.Counter()
            for sx in state_list:
                if not mdp.isTerminal(sx):
                    possible_action = mdp.getPossibleActions(sx)
                    arg_max = float("-inf")
                    for possible_action in possible_action:
                        transition_states_probs = mdp.getTransitionStatesAndProbs(sx, possible_action)
                        for transition_state, probability in transition_states_probs: #enumerate
                            updated_discount = (discount * self.values[transition_state])
                            reward = mdp.getReward(sx, possible_action, transition_state)
                            temp_score = updated_discount + reward
                            iter_score += temp_score * probability
                        arg_max = max(arg_max, iter_score)
                        label[sx] = arg_max
                        iter_score = 0
                else:
                    label[sx] = 0
            self.values = label
            i += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        sx = state
        transition_states_probs = self.mdp.getTransitionStatesAndProbs(sx, action)
        q_value = 0

        for transition_state, probability in transition_states_probs:
            updated_discount = self.discount * self.values[transition_state]
            reward = self.mdp.getReward(sx, action, transition_state)
            temp_q_value = reward + updated_discount
            q_value += (temp_q_value * probability)
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        arg_max = float("-inf")
        label = None
        omdp  = self.mdp
        sx = state
        possible_actions = omdp.getPossibleActions(sx)
        if not omdp.isTerminal(sx):
            for possible_action in possible_actions:
                temp_value = self.computeQValueFromValues(sx, possible_action)
                if arg_max <= temp_value:
                    arg_max = temp_value
                    label = possible_action
            return label
        else:
            return label

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
