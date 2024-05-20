# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.Qvalues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if(self.Qvalues[state, action]):
          return self.Qvalues[state, action]
        return 0.0


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if legalActions:
          max = self.getQValue(state, legalActions[0])
          for action in legalActions:
            qval = self.getQValue(state, action)
            if qval > max:
              max = qval
              
          return max
        return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if legalActions:
          bestAction = legalActions[0]
          max = self.getQValue(state, legalActions[0])
          for action in legalActions:
            qval = self.getQValue(state, action)
            if qval > max:
              max = qval
              bestAction = action
          orAct = bestAction
          for action in legalActions:
            qval = self.getQValue(state, action)
            if qval == max and action != orAct:
              bestAction = random.choice([bestAction, action])
          return bestAction
        return 'None'

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        prob = self.epsilon
        if legalActions:
            if util.flipCoin(prob):
              return random.choice(legalActions)
            action = legalActions[0]
            max = self.getQValue(state, legalActions[0])
            for a in legalActions:
              qval = self.getQValue(state, a)
              if qval > max:
                max = qval
                action = a
            orAct = action
            for a in legalActions:
              qval = self.getQValue(state, a)
              if qval == max and a != orAct:
                action = random.choice([action, a])
            return action
        return 'None'

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.Qvalues[state, action] = (1-self.alpha)*self.Qvalues[state, action]+self.alpha*(reward + self.discount*self.getValue(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        feats = self.featExtractor.getFeatures(state, action)
        qvalue = 0
        for feat in feats:
          qvalue += feats[feat]*self.weights[feat]
        return qvalue
        "*** YOUR CODE HERE ***"

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        feats = self.featExtractor.getFeatures(state, action)
        difference = (reward + self.discount*self.getValue(nextState)) - self.getQValue(state, action)
        for feat in feats:
          self.weights[feat] = self.weights[feat] + (self.alpha * difference)*feats[feat]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


# """
# TEAM PROJECT CODE
# """
# class Transition:
#   def __init__(self, state, action, reward, next_state):
#       self.state = state
#       self.action = action
#       self.reward = reward
#       self.next_state = next_state

# class Reinforce(ReinforcementAgent):
#     """
#        Custom agent for REINGFORCE with linear approximation using softmax
#     """
#     def __init__(self, extractor='IdentityExtractor', epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
#         self.featExtractor = util.lookup(extractor, globals())()
#         #PacmanQAgent.__init__(self, **args)
#         args['epsilon'] = epsilon
#         args['gamma'] = gamma
#         args['alpha'] = alpha
#         args['numTraining'] = numTraining
#         self.index = 0  # This is always Pacman
#         ReinforcementAgent.__init__(self, **args)
#         self.policy = util.Counter()
#         self.newpolicy = util.Counter()
#         self.episode = []

#     def getWeights(self):
#         return self.weights

#     def getSoftMax(self, state, action):
#         hA = 0
#         hB = 0
#         lActions = self.getLegalActions(state)
#         for b in lActions:
#           oFeats = self.featExtractor.getFeatures(state, b)
#           phhB = sum(self.policy[feat] * oFeats[feat] for feat in oFeats)
#           if b == action: hA = phhB
#           hB += math.exp(phhB)
#         #print(hA, hB, (math.exp(hA) / hB))
#         return (math.exp(hA) / hB)

#     def updatePolicy(self, state, action, advantage, t):
#       feats = self.featExtractor.getFeatures(state, action)
#       lActions = self.getLegalActions(state)
      
#       for feat in feats:
#         diff = sum(self.getSoftMax(state, b) * self.featExtractor.getFeatures(state, b)[feat] for b in lActions)
#         actionPolicy = feats[feat] - diff
#         self.newpolicy[feat] = self.policy[feat] + self.alpha * (self.discount**t) * advantage * actionPolicy 
#       for feat in feats:
#         self.policy[feat] = self.newpolicy[feat]

#     def getAction(self, state):
#       lActions = self.getLegalActions(state)
#       prob = [self.getSoftMax(state, a) for a in lActions]
#       #print(prob)
#       action = np.random.choice(lActions, p=prob)
#       self.doAction(state, action)
#       return action
      
#     def getValue(self, state):
#       lActions = self.getLegalActions(state)
#       prob = [self.getSoftMax(state, a) for a in lActions]
#       return np.random.choice(prob, p=prob)

#     def update(self, state, action, nextState, reward):
#         """
#            Should update your weights based on transition
#         """
#         "*** YOUR CODE HERE ***"
#         transition = Transition(state, action, reward, nextState)
#         self.episode.append(transition)

#         #util.raiseNotDefined()
#     """
#     def episodeUpdate(self):
#         #pseudo code
#         #G = sum(self.discount**i * reward for i, t in enumerate(episode[t:])
#         for t, transition in enumerate(self.episode):
#           G = sum(self.discount**i * t.reward for i, t in enumerate(self.episode[t:]))
#           baseline = 0
#           advantage = G - baseline
#           #update policy
#           self.updatePolicy(transition.state, transition.action, advantage)
#         self.episode = []
#     """

#     def final(self, state):
#         "Called at the end of each game."
#         # call the super-class final method
#         ReinforcementAgent.final(self, state)
        
#         #pseudo code
#         #G = sum(self.discount**i * reward for i, t in enumerate(episode[t:])
#         for it, transition in enumerate(self.episode):
#           G = sum(self.discount**i * t.reward for i, t in enumerate(self.episode[it:]))
#           #G = 
#           baseline = 0
#           advantage = G - baseline
#           #update policy
#           #print(advantage)
#           self.updatePolicy(transition.state, transition.action, advantage, it)
#         #self.policy = self.newpolicy
#         self.newpolicy = util.Counter()
#         self.episode = []
        
#         # did we finish training?
#         if self.episodesSoFar == self.numTraining:
#             # you might want to print your weights here for debugging
#             #for policy in self.policy:
#               #print(policy)
            
#             pass

class ReinforceAgent(PacmanQAgent):
    """
       ReinforceAgent implements a simple REINFORCE (Monte Carlo Policy Gradient) algorithm
       with function approximation.

        Instance variables used
        - weights
        - features
        - alpha
        - gamma
        - epsilon
        - episode reward
        - episode actions
        - episode states

    """
    def __init__(self, extractor='SimpleExtractor', **args):
        # Initialize the agent with provided parameters
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        # Initialize weights for each feature
        self.weights = util.Counter()

        # Initializong all weights to 0
        self.weights['bias'] = 0
        self.weights['#-of-ghosts-1-step-away'] = 0
        self.weights['eats-food'] = 0
        self.weights['closest-food'] = 0

        # Chance
        self.epsilon = 0.01
        
        # learning rate: α (Alpha)
        self.alpha = args.get('alpha', 0.01)
        # Discount factor: γ (Gamma)
        self.gamma = args.get('gamma', 0.9)  
        
        # Lists to store episode data
        self.EpiStates, self.EpiActions, self.EpiRewards = [], [], []

    def getAction(self, state):
        availableActions = self.getLegalActions(state)
        # Filter out "Stop" action
        actionSet = [action for action in availableActions if action != "Stop"] if availableActions else []
        actionProbs = self.getProbActions(state)    
        
        # Exploration vs. exploitation trade-off
        action = random.choice(actionSet) if (self.episodesSoFar < self.numTraining and util.flipCoin(self.epsilon)) else max(actionProbs, key=actionProbs.get)

        self.doAction(state, action)
        return action

    def getProbActions(self, state):
        availableActions = self.getLegalActions(state)
        # Filter out "Stop" action
        actionSet = [action for action in availableActions if action != "Stop"] if availableActions else []
        # Compute preferences for each action
        actionFeatureSum = util.Counter({action: sum(self.weights[f] * features[f] for f in features) for action in actionSet for features in [self.featExtractor.getFeatures(state, action)]})
        # Stable softmax computation
        e_values = {a: math.exp(actionFeatureSum[a]) for a in actionFeatureSum}
        total = sum(e_values.values())
        
        # Compute action actionProbs
        softmaxProbs = util.Counter({a: e_values[a] / total for a in e_values})
        return softmaxProbs

    def update(self, state, action, nextState, reward):
        """
           Update the weights based on the REINFORCE algorithm.
        """
        # Append current state, action, and reward to their respective episode lists
        self.EpiStates.append(state)
        self.EpiActions.append(action)
        self.EpiRewards.append(reward)
    
    def getGradient(self, state, action):
        availableActions = self.getLegalActions(state)
        features = self.featExtractor.getFeatures(state, action)
        ProbAction = self.getProbActions(state)

        # Compute expected features for all actions
        featureUpdate = util.Counter({f: sum(ProbAction[a] * a_features[f] for a in availableActions for a_features in [self.featExtractor.getFeatures(state, a)]) for f in features})

        # Compute gradient
        grad = util.Counter({f: features[f] - featureUpdate[f] for f in features})
        return grad

    def computeReturns(self, state, action, nextState, reward):
        """
           Compute the return (cumulative future rewards) from the current time step.
        """
        return reward

    def final(self, state):
        totalReturns, AccumulatedRewards = [], 0
        # Compute returns for each time step
        for reward in reversed(self.EpiRewards):
            AccumulatedRewards = reward + self.gamma * AccumulatedRewards
            totalReturns.insert(0, AccumulatedRewards)

        # Update weights using REINFORCE algorithm
        for t, (reward, state, action) in enumerate(zip(reversed(self.EpiRewards), self.EpiStates, self.EpiActions)):
            AccumulatedRewards = totalReturns[t]
            grad = self.getGradient(state, action)
            # Update weights
            for feature in grad:
                self.weights[feature] += self.alpha * AccumulatedRewards * grad[feature]

        # Clear episode data for the next episode
        self.EpiStates.clear()
        self.EpiActions.clear()
        self.EpiRewards.clear()

        # Perform final steps
        PacmanQAgent.final(self, state)

class ActorCriticAgent(PacmanQAgent):
    """
       ActorCriticAgent implements the Actor-Critic algorithm
       with function approximation.
    """
    def __init__(self, extractor='SimpleExtractor', **args):
        # Initialize the agent with provided parameters
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        # Initialize weights for actor and critic
        self.actorWeights = util.Counter()
        self.criticWeights = util.Counter()

        # Chance
        self.epsilon = 0.01

        # Set learning rates for actor and critic: α's (Alpha's)
        self.actorAlpha = args.get('actorAlpha', 0.01)
        self.criticAlpha = args.get('criticAlpha', 0.01)
        # Discount factor: γ (Gamma)
        self.gamma = args.get('gamma', 0.9)

        # Lists to store episode data
        self.EpiStates, self.EpiActions, self.EpiRewards = [], [], []

    def getAction(self, state):
        availableActions = self.getLegalActions(state)
        # Filter out "Stop" action
        actionSet = [action for action in availableActions if action != "Stop"] if availableActions else []
        actionProbs = self.getProbActions(state)    
        
        # Exploration vs. exploitation trade-off
        action = random.choice(actionSet) if (self.episodesSoFar < self.numTraining and util.flipCoin(self.epsilon)) else max(actionProbs, key=actionProbs.get)

        # Perform the chosen action
        self.doAction(state, action)
        return action

    def getProbActions(self, state):
        availableActions = self.getLegalActions(state)
        # Filter out "Stop" action
        actionSet = [action for action in availableActions if action != "Stop"] if availableActions else []
        # Compute preferences for each action
        actionFeatureSum = util.Counter({action: sum(self.actorWeights[f] * features[f] for f in features) for action in actionSet for features in [self.featExtractor.getFeatures(state, action)]})
        # Stable softmax computation
        e_values = {a: math.exp(actionFeatureSum[a]) for a in actionFeatureSum}
        total = sum(e_values.values())
        # Compute action probabilities
        softmaxProbs = util.Counter({a: e_values[a] / total for a in e_values})
        return softmaxProbs

    def update(self, state, action, nextState, reward):
        # Append current state, action, and reward to their respective episode lists
        self.EpiStates.append(state)
        self.EpiActions.append(action)
        self.EpiRewards.append(reward)

        # Compute error
        valueNewState = self.computeStateValue(nextState) if nextState is not None else 0
        valueState = self.computeStateValue(state)
        errorTempDiff = reward + self.gamma * valueNewState - valueState

        # Update critic weights
        for feature in self.criticWeights:
            self.criticWeights[feature] += self.criticAlpha * errorTempDiff * self.featExtractor.getFeatures(state)[feature]

        # Update actor weights
        grad = self.getGradient(state, action)
        for feature in grad:
            self.actorWeights[feature] += self.actorAlpha * errorTempDiff * grad[feature]

    def getGradient(self, state, action):
        availableActions = self.getLegalActions(state)
        features = self.featExtractor.getFeatures(state, action)
        ProbAction = self.getProbActions(state)

        # Compute expected features for all actions
        featureUpdate = util.Counter({f: sum(ProbAction[a] * a_features[f] for a in availableActions for a_features in [self.featExtractor.getFeatures(state, a)]) for f in features})

        # Compute gradient
        grad = util.Counter({f: features[f] - featureUpdate[f] for f in features})
        return grad

    def computeStateValue(self, state):
        """
           Compute the value of a state using the critic weights.
        """
        state_value = sum(self.criticWeights[f] * self.featExtractor.getFeatures(state)[f] for f in self.criticWeights)
        return state_value

    def final(self, state):
        """
           Perform final updates at the end of the episode.
        """
        # Clear episode data for the next episode
        self.EpiRewards.clear()
        self.EpiActions.clear()
        self.EpiStates.clear()

        # Perform final steps
        PacmanQAgent.final(self, state)
