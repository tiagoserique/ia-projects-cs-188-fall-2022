# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # if the next state is a win, return a high score
        if successorGameState.isWin():
            return 1000000 

        
        """ pacman variables """
        current_pos_pacman = currentGameState.getPacmanPosition()
        
        
        """ food variables """
        current_number_food_left  = len(currentGameState.getFood().asList())
        sucessor_number_food_left = len(newFood.asList())
        sucessor_number_pellets   = len(successorGameState.getCapsules())
        current_nearest_food      = min([
            manhattanDistance(current_pos_pacman, food) 
                for food in currentGameState.getFood().asList()
        ])
        successor_nearest_food = min([
            manhattanDistance(newPos, food) for food in newFood.asList()
        ])
        successor_nearest_food = successor_nearest_food if successor_nearest_food else 0


        """ ghost variables """
        sucessor_sum_scared_times = sum(newScaredTimes)
        current_min_distance_ghost = min([
            manhattanDistance(newPos, ghost.getPosition()) 
                for ghost in currentGameState.getGhostStates()
        ])
        sucessor_min_distance_ghost = min([
            manhattanDistance(newPos, ghost.getPosition())
                for ghost in newGhostStates
        ])


        """ determine score """
        # relative score to the current state score
        score = successorGameState.getScore() - currentGameState.getScore()

        # increase score if the next state is closer to the nearest food
        if successor_nearest_food < current_nearest_food:
            score += 400

        # increase score if the next state is closer to the nearest ghost
        if sucessor_number_food_left < current_number_food_left:
            score += 200
        else:
            score -= 100

        # decrease score for each food left
        score -= 10 * sucessor_number_food_left

        # increase score if the next action is the same as the current direction
        direction = currentGameState.getPacmanState().getDirection()
        if direction == action:
            score += 5

        # decrease score if the next action is STOP
        if action == Directions.STOP:
            score -= 10

        # increase score if the next state is eating a pellet
        if newPos in currentGameState.getCapsules():
            score += 150 * sucessor_number_pellets

        # if the ghost is scared, more close to him is better
        if sucessor_sum_scared_times > 0:
            if current_min_distance_ghost < sucessor_min_distance_ghost:
                score += 50
            else:
                score -= 10
        # if the ghost is not scared, more far from him is better
        else:
            if current_min_distance_ghost > sucessor_min_distance_ghost:
                score += 50
            else:
                score -= 50

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agent_index, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        import sys

        def max_level(game_state, depth):
            current_depth = depth + 1
            
            condition = game_state.isWin() or game_state.isLose() or current_depth == self.depth
            if condition: 
                return self.evaluationFunction(game_state)
            
            max_value = -9999999
            actions = game_state.getLegalActions(0)
            
            for action in actions:
                successor = game_state.generateSuccessor(0, action)
                max_value = max(
                    max_value, 
                    min_level(successor, current_depth, 1)
                )
            
            return max_value
        

        def min_level(game_state, depth, agent_index):
            min_value = sys.maxsize
            
            if game_state.isWin() or game_state.isLose():
                return self.evaluationFunction(game_state)
            
            actions = game_state.getLegalActions(agent_index)
            
            for action in actions:
                successor = game_state.generateSuccessor(agent_index, action)
                if agent_index == (game_state.getNumAgents() - 1):
                    min_value = min(min_value, max_level(successor, depth))
                else:
                    min_value = min(min_value, min_level(successor, depth, agent_index + 1))
            
            return min_value
     
        actions = gameState.getLegalActions(0)
        current_score = -9999999
        return_action = ''
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)

            score = min_level(next_state, 0, 1)

            if score > current_score:
                return_action = action
                current_score = score
        
        return return_action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        import sys

        def max_level(game_state, depth, alpha, beta):
            current_depth = depth + 1

            condition = game_state.isWin() or game_state.isLose() or current_depth == self.depth 
            if condition: 
                return self.evaluationFunction(game_state)

            max_value = -9999999
            actions   = game_state.getLegalActions(0)
            new_alpha = alpha

            for action in actions:
                successor = game_state.generateSuccessor(0, action)
                max_value = max(
                    max_value, 
                    min_level(successor, current_depth, 1, new_alpha, beta)
                )
                
                if max_value > beta:
                    return max_value
                
                new_alpha = max(new_alpha, max_value)

            return max_value
        
        def min_level(game_state, depth, agent_index, alpha, beta):
            min_value = sys.maxsize

            if game_state.isWin() or game_state.isLose(): 
                return self.evaluationFunction(game_state)

            actions  = game_state.getLegalActions(agent_index)
            new_beta = beta
            for action in actions:
                successor = game_state.generateSuccessor(agent_index, action)

                if agent_index == (game_state.getNumAgents() - 1):
                    min_value = min(
                        min_value, 
                        max_level(successor, depth, alpha, new_beta)
                    )

                    if min_value < alpha:
                        return min_value
                    
                    new_beta = min(new_beta, min_value)
                else:
                    min_value = min(
                        min_value, 
                        min_level(
                            successor, 
                            depth, 
                            agent_index + 1, 
                            alpha, 
                            new_beta
                        )
                    )

                    if min_value < alpha:
                        return min_value
                    
                    new_beta = min(new_beta, min_value)

            return min_value

        actions = gameState.getLegalActions(0)
        current_score = -9999999
        return_action = ''
        alpha = -9999999
        beta  = 9999999
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            score = min_level(nextState, 0, 1, alpha, beta)

            if score > current_score:
                return_action = action
                current_score = score

            if score > beta:
                return return_action

            alpha = max(alpha, score)

        return return_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def max_level(game_state, depth):
            current_depth = depth + 1

            condition = game_state.isWin() or game_state.isLose() or current_depth == self.depth
            if condition: 
                return self.evaluationFunction(game_state)

            max_value = -9999999
            actions = game_state.getLegalActions(0)
            for action in actions:
                successor = game_state.generateSuccessor(0, action)
                max_value = max(
                    max_value, 
                    expect_level(successor, current_depth, 1)
                )

            return max_value
        
        def expect_level(game_state, depth, agent_index):
            if game_state.isWin() or game_state.isLose(): 
                return self.evaluationFunction(game_state)

            actions = game_state.getLegalActions(agent_index)
            total_expected_value = 0
            number_of_actions    = len(actions)
            for action in actions:
                successor = game_state.generateSuccessor(agent_index, action)

                if agent_index == (game_state.getNumAgents() - 1):
                    expected_value = max_level(successor, depth)
                else:
                    expected_value = expect_level(
                        successor, 
                        depth, 
                        agent_index + 1
                    )

                total_expected_value = total_expected_value + expected_value

            if number_of_actions == 0:
                return  0

            return float(total_expected_value)/float(number_of_actions)
        
        actions = gameState.getLegalActions(0)
        currentScore = -9999999
        returnAction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            score = expect_level(nextState, 0, 1)

            if score > currentScore:
                returnAction = action
                currentScore = score
                
        return returnAction

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    score = 0
    new_position = currentGameState.getPacmanPosition()
    new_food     = currentGameState.getFood()
    new_ghost_states   = currentGameState.getGhostStates()
    number_of_pellets  = len(currentGameState.getCapsules())
    number_of_no_foods = len(new_food.asList(False))           
    new_scared_times   = [
        ghost_state.scaredTimer 
        for ghost_state in new_ghost_states
    ]


    ghost_positions = []
    for ghost in new_ghost_states:
        ghost_positions.append(ghost.getPosition())
    
    ghost_distance = [0]
    for position in ghost_positions:
        ghost_distance.append(manhattanDistance(new_position, position))


    food_list = new_food.asList()
    food_distance = [0]
    for position in food_list:
        food_distance.append(manhattanDistance(new_position, position))


    sum_scared_times   = sum(new_scared_times)
    sum_ghost_distance = sum (ghost_distance)
    reciprocal_food_distance = 0
    if sum(food_distance) > 0:
        reciprocal_food_distance = 1.0 / sum(food_distance)
        
    score += currentGameState.getScore() + reciprocal_food_distance + number_of_no_foods

    if sum_scared_times > 0:    
        score += sum_scared_times + (-1 * number_of_pellets) + (-1 * sum_ghost_distance)
    else:
        score += sum_ghost_distance + number_of_pellets
    
    return score

# Abbreviation
better = betterEvaluationFunction
