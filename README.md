# QLearning Tic-Tac-Toe Bot

This project is an attempt at using a neural network (from my Cpp-Neural-Net project) to learn to play Tic-Tac-Toe using a Q-Learning approach. 

## Method

### Network Architecture

#### Input Layer

The input layer consists of 27 neurons. This is because there are 3 states (X, O, or blank) for each of the 9 positions on the Tic-Tac-Toe board. The input array is fed into the network such that the first 9 places are the positions of the X's, the second 9 are the position of the O's, and the last 9 are the position of the blanks.
<p>
  <img src="/imgs/ExampleBoard.png" width="250" height="350" align="left">
  Corresponding input: <img src="/imgs/ExampleInput.png" width="350" height="75" align="right">
</p>

#### Hidden Layers

There is only one hidden layer that consists of 243 neurons (9 times the number of inputs), which are fully connected to the input layer. The activation function for these neurons is the [Softplus ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus).

#### Output Layer

The output layer consists of 9 neurons, one for each position of the board, which are fully connected to the hidden layer. The activation function for these neurons is Linear, so the network can approximate the Q-Value of making a move at each position.

The gradient descent optimizer chosen for training was [ADAM](https://github.com/drew-english/QLearning-Tic-Tac-Toe-Bot/blob/e50e1a24c950c31653f7213132fd029d4d23f2dd/src/NeuralNet.cpp#L36-L45), as it proved to be better for finding a minima during gradient descent than [RMSProp](https://github.com/drew-english/QLearning-Tic-Tac-Toe-Bot/blob/e50e1a24c950c31653f7213132fd029d4d23f2dd/src/NeuralNet.cpp#L28-L34). The hyperparameters can be found [here](https://github.com/drew-english/QLearning-Tic-Tac-Toe-Bot/blob/e50e1a24c950c31653f7213132fd029d4d23f2dd/lib/NeuralNet.h#L23-L27).


### Training the Network

#### The Competition

To train the network we need something to play against! In comes the Random player and the Min-Max player. The [Random player](https://github.com/drew-english/QLearning-Tic-Tac-Toe-Bot/blob/master/src/RANDPlayer.cpp) follows its name and picks a random open position on the board each time. However, the [Min-Max player](https://github.com/drew-english/QLearning-Tic-Tac-Toe-Bot/blob/master/src/MINMAXPlayer.cpp) is a little smarter than the Random player. It assigns a score to each possible move by exploring every move available left in the game, and it will get these scores after it reaches an end-game state. After an end-game state is reached, the scores (1 for win, -1 for loss, and 0 for draw) are propagated back up to the original state (assuming we will choose the move to maximize the score and the other player will choose the move that minimizes our score) where we can then make our move. This process is inefficient as every possible move needs to be explored, but it makes the best move possible every time. A perfect player.
To make the Min-Max player more efficient we stored moves in a hash map, so if it found itself in a state it has already been in, it has a list of best moves it can make.

#### Creating a Versatile Player

To train a network so that it could play well versus a human, it needs to be trained such that it can play well against a diverse set of strategies. This is where the competition steps in, we have a Random player to train the network how to play against odd strategies, and a Min-Max player for learning to play against optimal strategies.

#### Testing
All performance tests were conducted by observing learning over 100 epochs, each epoch containing 100 games. Many tests were conducted before finding settling on the hyperparameters (listed above) for the network. Tests were conducted for the network player going first and second, versus both the Random player and the Min-Max player.


## Results

Display graphs for learning vs Random player and learning vs Min-Max player

## Authors

* **Drew English** - [DrewEnglish](https://github.com/drew-english)

## Acknowledgments
