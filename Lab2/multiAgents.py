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
import random, util, sys

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition()      # Pacman position after moving
        newFood = successorGameState.getFood()               # Remaining food
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        listFood = newFood.asList()                        # All remaining food as list
        ghostPos = successorGameState.getGhostPositions()  # Get the ghost position
        # Initialize with list 
        mFoodDist = []
        mGhostDist = []

        # Find the distance of all the foods to the pacman 
        for food in listFood:
          mFoodDist.append(manhattanDistance(food, newPos))

        # Find the distance of all the ghost to the pacman
        for ghost in ghostPos:
          mGhostDist.append(manhattanDistance(ghost, newPos))

        if currentGameState.getPacmanPosition() == newPos:
          return (-(float("inf")))

        for ghostDistance in mGhostDist:
          if ghostDistance < 2:
            return (-(float("inf")))

        if len(mFoodDist) == 0:
          return float("inf")
        else:
          minFoodDist = min(mFoodDist)
          maxFoodDist = max(mFoodDist)

        return 1000/sum(mFoodDist) + 10000/len(mFoodDist)


def scoreEvaluationFunction(currentGameState):
    """
    This evaluation function returns the score of the current game state, considering:
    - Distance to the nearest food (pallets)
    - Distance to the nearest ghost (danger)
    - Number of remaining food pallets
    """

    # Pacman's current position
    pacmanPos = currentGameState.getPacmanPosition()

    # Get remaining food
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()

    # Get positions and states of the ghosts
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()

    # --- Calculate Pallet Score ---
    foodDistances = [manhattanDistance(pacmanPos, food) for food in foodList]
    if len(foodDistances) > 0:
        minFoodDist = min(foodDistances)  # Nearest food
        palletScore = 1.0 / minFoodDist   # Closer food = higher score
    else:
        palletScore = float('inf')        # If no food left, maximum score (winning state)
    
    # --- Calculate Ghost Danger ---
    ghostDistances = [manhattanDistance(pacmanPos, ghostPos) for ghostPos in ghostPositions]
    ghostDanger = 0
    for ghostDist in ghostDistances:
        if ghostDist > 0:  # Avoid division by zero
            ghostDanger += 1.0 / ghostDist  # Closer ghost = higher danger

    # --- Add bonus for current game score ---
    currentScore = currentGameState.getScore()

    # --- Final Score Calculation ---
    finalScore = currentScore + palletScore - ghostDanger

    return finalScore



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

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


import time

class MinimaxAgent(MultiAgentSearchAgent):
    """
    MiniMax agent with Progressive Deepening and Move Ordering improvements.
    """

    def getAction(self, gameState):
        """
        Returns the best action using MiniMax with Progressive Deepening and Move Ordering.
        """
        def minimax(agentIndex, depth, gameState, alpha, beta):
            # Base case: terminal state or depth limit reached
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Pacman's turn (Maximizing)
            if agentIndex == 0:
                return max_value(agentIndex, depth, gameState, alpha, beta)

            # Ghost's turn (Minimizing)
            else:
                return min_value(agentIndex, depth, gameState, alpha, beta)

        def max_value(agentIndex, depth, gameState, alpha, beta):
            maxScore = float('-inf')
            legalActions = self.getOrderedLegalActions(gameState, agentIndex)  # Use ordered actions

            if not legalActions:
                return self.evaluationFunction(gameState)

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = minimax(1, depth, successor, alpha, beta)
                maxScore = max(maxScore, score)

                if maxScore >= beta:  # Beta cutoff
                    return maxScore
                alpha = max(alpha, maxScore)

            return maxScore

        def min_value(agentIndex, depth, gameState, alpha, beta):
            minScore = float('inf')
            legalActions = self.getOrderedLegalActions(gameState, agentIndex)  # Use ordered actions

            if not legalActions:
                return self.evaluationFunction(gameState)

            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = minimax(nextAgent, nextDepth, successor, alpha, beta)
                minScore = min(minScore, score)

                if minScore <= alpha:  # Alpha cutoff
                    return minScore
                beta = min(beta, minScore)

            return minScore

        # Progressive Deepening: Start with depth 1 and incrementally deepen the search
        bestAction = None
        startTime = time.time()
        maxDepth = self.depth  # The maximum depth we want to reach
        currentDepth = 1
        timeLimit = 2  # Set a reasonable time limit (in seconds) for searching

        while currentDepth <= maxDepth:
            legalActions = gameState.getLegalActions(0)
            bestScore = float('-inf')

            # Iterating through actions with the current depth
            for action in legalActions:
                successor = gameState.generateSuccessor(0, action)
                score = minimax(1, currentDepth, successor, float('-inf'), float('inf'))
                if score > bestScore:
                    bestScore = score
                    bestAction = action

            # Check if we are out of time
            if time.time() - startTime > timeLimit:
                break  # Return the best action found so far

            currentDepth += 1  # Increase the depth for the next iteration

        return bestAction

    def getOrderedLegalActions(self, gameState, agentIndex):
            """
            Returns legal actions ordered based on the evaluation function (Move Ordering).
            """
            legalActions = gameState.getLegalActions(agentIndex)
            actionScores = []

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = self.evaluationFunction(successor)
                actionScores.append((score, action))

            # Sort actions by their evaluation scores in descending order
            actionScores.sort(reverse=True, key=lambda x: x[0])
            orderedActions = [action for score, action in actionScores]

            return orderedActions


# class MinimaxAgent(MultiAgentSearchAgent):
#     def getAction(self, gameState):
#         """
#         Returns the minimax action using the given depth and evaluation function.
#         """

#         def minimax(agentIndex, depth, gameState):
#             # Base case: terminal state or depth limit reached
#             if depth == 0 or gameState.isWin() or gameState.isLose():
#                 return self.evaluationFunction(gameState)

#             # Pacman's turn (maximizing)
#             if agentIndex == 0:
#                 return max_value(agentIndex, depth, gameState)

#             # Ghost's turn (minimizing)
#             else:
#                 return min_value(agentIndex, depth, gameState)

#         # Pacman’s turn (initial call)
#         bestAction = None
#         bestScore = float('-inf')

#         for action in gameState.getLegalActions(0):
#             successor = gameState.generateSuccessor(0, action)
#             score = minimax(1, self.depth, successor)
#             if score > bestScore:
#                 bestScore = score
#                 bestAction = action

#         return bestAction

# def max_value(agentIndex, depth, gameState):
#     maxScore = float('-inf')
#     for action in gameState.getLegalActions(agentIndex):
#         successor = gameState.generateSuccessor(agentIndex, action)
#         score = minimax(1, depth, successor)
#         maxScore = max(maxScore, score)
#     return maxScore

# def min_value(agentIndex, depth, gameState):
#     minScore = float('inf')
#     for action in gameState.getLegalActions(agentIndex):
#         successor = gameState.generateSuccessor(agentIndex, action)
#         score = minimax(1, depth, successor)
#         minScore = min(minScore, score)
#     return minScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Alpha-Beta Pruning agent implementation.
    """

    def getAction(self, gameState):
        """
          Returns the action using Alpha-Beta Pruning.
        """
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            # Base case: terminal state or depth limit reached
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # Pacman's turn (Maximizing)
            if agentIndex == 0:
                return max_value(agentIndex, depth, gameState, alpha, beta)
            
            # Ghost's turn (Minimizing)
            else:
                return min_value(agentIndex, depth, gameState, alpha, beta)

        def max_value(agentIndex, depth, gameState, alpha, beta):
            maxScore = float('-inf')
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = alphaBeta(1, depth, successor, alpha, beta)
                maxScore = max(maxScore, score)
                
                if maxScore >= beta:  # Beta cutoff (prune)
                    return maxScore
                alpha = max(alpha, maxScore)  # Update alpha

            return maxScore

        def min_value(agentIndex, depth, gameState, alpha, beta):
            minScore = float('inf')
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = alphaBeta(nextAgent, nextDepth, successor, alpha, beta)
                minScore = min(minScore, score)

                if minScore <= alpha:  # Alpha cutoff (prune)
                    return minScore
                beta = min(beta, minScore)  # Update beta

            return minScore

        # Initialize alpha (-inf) and beta (+inf) for Alpha-Beta pruning
        alpha = float('-inf')
        beta = float('inf')

        # Find the best action for Pacman (agentIndex = 0)
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestScore = float('-inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = alphaBeta(1, self.depth, successor, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

import heapq

def aStarSearch(gameState, start, goalPositions, heuristic):
    """
    A* search algorithm to find the shortest path from the start to the goal.
    :param gameState: The current game state
    :param start: Pacman's starting position
    :param goalPositions: List of goal positions (e.g., food)
    :param heuristic: Heuristic function (e.g., Manhattan distance)
    :return: A list of actions that lead to the nearest goal
    """

    # Priority queue for A* (stores tuples of (cost, position, path))
    frontier = []
    heapq.heappush(frontier, (0, start, []))  # (priority, current position, path)

    # Set for visited positions
    visited = set()

    while frontier:
        cost, currentPos, path = heapq.heappop(frontier)

        # Check if we've reached the goal
        if currentPos in goalPositions:
            return path  # Return the path to the goal

        # If not yet visited
        if currentPos not in visited:
            visited.add(currentPos)

            # Explore neighbors
            for action in gameState.getLegalActions(0):  # Pacman's legal actions
                successor = gameState.generateSuccessor(0, action)
                nextPos = successor.getPacmanPosition()

                if nextPos not in visited:
                    newCost = cost + 1  # Increment cost
                    newPath = path + [action]  # Update path
                    priority = newCost + heuristic(nextPos, goalPositions)
                    heapq.heappush(frontier, (priority, nextPos, newPath))

    return []  # No path found

def manhattanHeuristic(position, goalPositions):
    """
    Heuristic for A* search based on Manhattan distance to the nearest goal.
    :param position: Pacman's current position
    :param goalPositions: List of goal positions (e.g., food or capsule)
    :return: Manhattan distance to the nearest goal
    """
    return min([manhattanDistance(position, goal) for goal in goalPositions])


class AStarMinimaxAgent(MultiAgentSearchAgent):
    """
    MiniMax agent with A* pathfinding improvement.
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction, with A* pathfinding for Pacman.
        """
        def minimax(agentIndex, depth, gameState):
            # Base case: terminal state or depth limit reached
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Pacman's turn (Maximizing)
            if agentIndex == 0:
                return max_value(agentIndex, depth, gameState)

            # Ghost's turn (Minimizing)
            else:
                return min_value(agentIndex, depth, gameState)

        def max_value(agentIndex, depth, gameState):
            maxScore = float('-inf')
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = minimax(1, depth, successor)
                maxScore = max(maxScore, score)

            return maxScore

        def min_value(agentIndex, depth, gameState):
            minScore = float('inf')
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = minimax(nextAgent, nextDepth, successor)
                minScore = min(minScore, score)

            return minScore

        # Use A* to determine the best path, but not to automatically pick the first action
        pacmanPos = gameState.getPacmanPosition()
        foodList = gameState.getFood().asList()

        # Call A* to get the best path to the nearest food
        bestPath = aStarSearch(gameState, pacmanPos, foodList, manhattanHeuristic)

        if bestPath:  # If A* found a path, take the first action in consideration
            bestAStarAction = bestPath[0]
            # Get the successor state for this action
            bestAStarSuccessor = gameState.generateSuccessor(0, bestAStarAction)
            bestAStarScore = minimax(1, self.depth, bestAStarSuccessor)
        else:
            bestAStarAction = None
            bestAStarScore = float('-inf')

        # Fallback to MiniMax if no A* path is found or MiniMax finds a better option
        legalActions = gameState.getLegalActions(0)
        bestAction = bestAStarAction
        bestScore = bestAStarScore

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, self.depth, successor)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction




class AStarAlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta Pruning agent with A* pathfinding improvement.
    """

    def getAction(self, gameState):
        """
        Returns the best action using Alpha-Beta Pruning with A* pathfinding.
        """
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman's turn (Maximizing)
                return max_value(agentIndex, depth, gameState, alpha, beta)
            else: 
                return min_value(agentIndex, depth, gameState, alpha, beta)

        def max_value(agentIndex, depth, gameState, alpha, beta):
            maxScore = float('-inf')
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = alphaBeta(1, depth, successor, alpha, beta)
                maxScore = max(maxScore, score)

                if maxScore >= beta:  # Beta cutoff
                    return maxScore
                alpha = max(alpha, maxScore)

            return maxScore

        def min_value(agentIndex, depth, gameState, alpha, beta):
            minScore = float('inf')
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = alphaBeta(nextAgent, nextDepth, successor, alpha, beta)
                minScore = min(minScore, score)

                if minScore <= alpha:  # Alpha cutoff
                    return minScore
                beta = min(beta, minScore)

            return minScore

        # Use A* to determine the best path, but not to automatically pick the first action
        pacmanPos = gameState.getPacmanPosition()
        foodList = gameState.getFood().asList()

        # Call A* to get the best path to the nearest food
        bestPath = aStarSearch(gameState, pacmanPos, foodList, manhattanHeuristic)

        if bestPath:  # If A* found a path, take the first action in consideration
            bestAStarAction = bestPath[0]
            bestAStarSuccessor = gameState.generateSuccessor(0, bestAStarAction)
            bestAStarScore = alphaBeta(1, self.depth, bestAStarSuccessor, float('-inf'), float('inf'))
        else:
            bestAStarAction = None
            bestAStarScore = float('-inf')

        # Fallback to Alpha-Beta if no A* path is found or Alpha-Beta finds a better option
        legalActions = gameState.getLegalActions(0)
        bestAction = bestAStarAction
        bestScore = bestAStarScore

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = alphaBeta(1, self.depth, successor, float('-inf'), float('inf'))
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction




def betterEvaluationFunction(currentGameState):
    """
      A more complex evaluation function considering:
      - Distance to the nearest food (pallet)
      - Distance to the nearest ghost (danger)
      - Number of remaining food pallets
      - Presence of power capsules
      - Ghost vulnerability (scared time)
    """
    # Pacman's position
    pacmanPos = currentGameState.getPacmanPosition()
    
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()

    # Get ghost states and positions
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()
    
    # --- Base Score (current game score) ---
    score = currentGameState.getScore()

    # --- Food Distance (Pallet Score) ---
    foodDistances = [manhattanDistance(pacmanPos, food) for food in foodList]
    if foodDistances:
        score += 1.0 / min(foodDistances)  # Closer food = higher score

    # --- Capsule Score ---
    capsuleDistances = [manhattanDistance(pacmanPos, capsule) for capsule in capsuleList]
    if capsuleDistances:
        score += 1.0 / min(capsuleDistances)  # Closer to power capsule = higher score

    # --- Ghost Distance and Scared Ghosts ---
    ghostDanger = 0
    scaredGhostBonus = 0
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDist = manhattanDistance(pacmanPos, ghostPos)

        if ghost.scaredTimer > 0:  # Ghost is scared
            scaredGhostBonus += 200 / ghostDist if ghostDist > 0 else 200
        else:  # Ghost is not scared, Pacman should avoid it
            if ghostDist > 0:
                ghostDanger += 1.0 / ghostDist  # Closer ghost = higher danger

    # Add scared ghost bonus and penalize ghost proximity
    score += scaredGhostBonus
    score -= ghostDanger

    # --- Remaining Food Penalty ---
    score -= len(foodList) * 10  

    # --- Capsule Bonus ---
    score -= len(capsuleList) * 50 

    return score



# Abbreviation
better = betterEvaluationFunction

