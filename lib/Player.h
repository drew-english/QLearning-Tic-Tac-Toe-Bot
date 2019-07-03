#ifndef PLAYER_H
#define PLAYER_H

#include "./NeuralNet.h"
#include "./TicTacToe.h"
#include <map>

//Defines a transition, used in experience replay for learning
struct Transition
{
    vector<double> state;
    int action;
    int reward;
    vector<double> qvals;
    vector<double> nextState;
    double nextMax = 0;
};


// Base player class, allows for polymorphism
class Player {
public:
    virtual void move(TicTacToe &game) = 0;
    vector<int> get_moves(TicTacToe &game); // required by players to see the valid moves of a game state
};


// player class, using a network for moves
class NNPlayer: public Player {
public:
    NNPlayer(Network *net, bool training = false, int minibatchSize = 16, int replaySize = 512); // constructor
    ~NNPlayer(); // destructor

    void move(TicTacToe &game); // routes move based on training
    void final_reward(int outcome); //writes the reward of the move which led to the game outcome (1 is win, -1 is loss, 0 is draw)
    vector<double> get_qvals(TicTacToe &game); // see the output of the network for testing 

private:
    Network *net;
    bool training;

    int rewardWin, rewardLoss, rewardValidMove, rewardDraw;
    double eps, epsMin; // for e-greedy action selection
    int minibatchSize, replaySize;
    int moves; // number of total moves the network has made
    double gamma; // discount rate of future rewards
    vector<Transition> replays, miniBatch; // for experience replay and sampling of replays

    //private functions:
    vector<double> get_input(TicTacToe &game); // changes game board into input for the network

    //returns qvals, but invalid moves are set to a small num. Also populates the possible moves
    vector<double> get_probs(TicTacToe &game, vector<double> probs, vector<int> &possibleMoves);
    int argmax(vector<double> const &x); // returns the position of the largest number in the vector
    int execute_move(TicTacToe &game, int action); // makes a move and returns the reward of the move
    void network_fit(vector<Transition> const &miniBatch); // updates the network based on a minibatch of transitions
    void train_move(TicTacToe &game); // Network move for training
    void notrain_move(TicTacToe &game); // Network move for normal game
};


//player class making random moves
class RANDPlayer: public Player {
public:
    RANDPlayer();
    ~RANDPlayer();

    void move(TicTacToe &game); // makes a move at random based on available moves
};


//MinMax player which finds the best possible move for a given board (assuming the other player will also choose their best move)
// for training against the network
class MINMAXPlayer: public Player {
public:
    MINMAXPlayer();
    ~MINMAXPlayer();

    void move(TicTacToe &game);
private:
    int valWin, valLoss, valDraw, side; // 0 is this players side, 1 is other players side
    
    //key is game board, val is the best moves for the game board
    std::map<vector<int>, vector<vector<int>>> moveCache; // caching values of boards so we do not need to repeat finding values

    //value finding functions
    vector<int> max(TicTacToe game);
    vector<int> min(TicTacToe game);
};

#endif