# MultiAgentPacman
Pacman AI 😎

**Multi Agent Pacman** is another version of pacman agent that will find its path with the minimax, alpha beta pruning, and expectimax to collect its foods, and the ghost while blinking.

You can use the commands below to run the project 

# List of commands Pacman AI
```
// For classic pacman game (you play)
python pacman.py 
// Easy 
python pacman.py -p ReflexAgent -l testClassic
// Reflex Agent with one ghost / two ghost 
python pacman.py --frameTime 0 -p ReflexAgent -k 1
python pacman.py --frameTime 0 -p ReflexAgent -k 2
// Minimax
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
// Alpha beta Agent on small map
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
```

# Video overview
<img src='http://i.imgur.com/IEpdZWC.gif' title='pacman' width='' alt='Video Walkthrough' />

The above shows the pacman behavior on reflex agent with two ghost.

