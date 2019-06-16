#include "../lib/Player.h"

vector<int> Player::get_moves(TicTacToe &game){
    vector<int> board = game.getBoard();
    vector<int> possibleMoves;

    for (int i = 0; i < board.size(); i++){
        if (board[i] == 2)
            possibleMoves.push_back(i);
    }

    return possibleMoves;
}

//NNPlayer functions:

NNPlayer::NNPlayer(Network *net, bool training, int minibatchSize, int replaySize){
    //init all vars
    this->training = training;
    this->minibatchSize = minibatchSize;
    this->replaySize = replaySize;
    
    this->moves = 0;
    this->eps = 1.0;
    this->epsMin = .1;
    this->gamma = .9;
    this->rewardWin = 30;
    this->rewardLoss = -30;
    this->rewardValidMove = 0;
    this->rewardDraw = 5;
    this->miniBatch = vector<Transition>(minibatchSize);
    this->net = net;
}

NNPlayer::~NNPlayer(){} // no mem manually allocated -> no mem to manually unallocate

//routes to the proper move
void NNPlayer::move(TicTacToe &game)
{ training ? train_move(game) : notrain_move(game); }


//Gives a normalized network input for a given game
// First 9 are position of X's, second 9 are positions of O's, and last 9 are position of blanks
vector<double> NNPlayer::get_input(TicTacToe &game){
    vector<int> board = game.getBoard();
    vector<double> input(27, 0);

    for (int i = 0; i < 9; i++){
        switch (board[i]){
        case 0:
            input[9 + i] = 1;
            break;
        case 1:
            input[i] = 1;
            break;
        case 2:
            input[18 + i] = 1;
            break;
        default:
            break;
        }
    }

    return input;
}


//returns the probs and updates possible moves (same loop for both)
vector<double> NNPlayer::get_probs(TicTacToe &game, vector<double> probs, vector<int> &possibleMoves){
    vector<int> board = game.getBoard();
    
    for(int i = 0; i < board.size(); i++){
        if(board[i] != 2)
            probs[i] = -5000; // sets prob of invalid move to something small so we do not select invalid moves 
        else
            possibleMoves.push_back(i);
    }

    return probs;
}


//returns the position of the largest value in the vector
int NNPlayer::argmax(vector<double> const &x){
    double max = INT64_MIN;
    int pos = 0;
    
    for(int i = 0; i < x.size(); i++){
        if (x[i] > max){
            max = x[i];
            pos = i;
        }
    }

    return pos;
}


// makes a move and returns the reward for that move
int NNPlayer::execute_move(TicTacToe &game, int action){
    int reward;

    if(game.makeMove(action + 1)){
        reward = this->rewardValidMove;
    }

    return reward;
}


void NNPlayer::network_fit(vector<Transition> const &miniBatch){
    vector<vector<double>> input, tout;
    vector<double> target;

    for (int i = 0; i < miniBatch.size(); i++){
        Transition t = miniBatch[i];
        input.push_back(t.state);

        target = t.qvals;
        target[t.action] = t.reward == this->rewardWin ? t.reward : t.reward + this->gamma * t.nextMax; // terminal state if reward is rewardWin
        tout.push_back(target);
    }

    net->batch_fit(input, tout);
}


void NNPlayer::train_move(TicTacToe &game){
    int action, reward;
    vector<double> state, newState, qvals, probs;
    vector<int> possibleMoves;

    state = get_input(game);
    qvals = this->net->run(state);
    probs = get_probs(game, qvals, possibleMoves); // favors qvals of valid moves to not waste time training on invalid moves

    //Select action using e-greedy:
    if ((double)rand() / (double)RAND_MAX < eps)
        action = possibleMoves[rand() % possibleMoves.size()]; // picks random move from possible moves
    else
        action = argmax(probs);

    //Take action, observe reward and new state:
    reward = execute_move(game, action);
    newState = get_input(game);

    //Change last move's next max to the current max of possible moves
    if(replays.size() > 0)
        this->replays[(moves - 1) % replaySize].nextMax = qvals[action];

    //Store new transition in replay:
    Transition t = {state, action, reward, qvals, newState};
    if (replays.size() < replaySize)
        this->replays.push_back(t);
    else { // replays is full, so rewrite old ones
        this->replays[moves % replaySize] = t;
    }

    //Sample mini batch of transitions from replays, then fit to the network:
    if (this->replays.size() >= this->minibatchSize)
    {
        for (int k = 0; k < minibatchSize; k++)
            this->miniBatch[k] = this->replays[rand() % replays.size()];

        network_fit(miniBatch);
    }

    this->moves++;
    if (this->eps > this->epsMin) // diminishes to epsMin over moves
        eps -= (1 - epsMin) / 5000;
}


//picks the move with the highest qval based on the state
void NNPlayer::notrain_move(TicTacToe &game){
    vector<double> qvals, probs;
    vector<int> possibleMoves;

    qvals = this->net->run(get_input(game));
    probs = get_probs(game, qvals, possibleMoves);
    game.makeMove(argmax(probs) + 1);
}

void NNPlayer::final_reward(int outcome){
    switch(outcome){
        case 1:
            replays[(moves - 1) % replaySize].reward = this->rewardWin;
            break;
        case 0:
            replays[(moves - 1) % replaySize].reward = this->rewardDraw;
            break;
        case -1:
            replays[(moves - 1) % replaySize].reward = this->rewardLoss;
            break;
        default:
            break;
    }
}


vector<double> NNPlayer::get_qvals(TicTacToe &game){
    return net->run(get_input(game));
}