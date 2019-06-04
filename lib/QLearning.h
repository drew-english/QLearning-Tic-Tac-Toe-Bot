#ifndef QLEARNING_H
#define QLEARNING_H

#include "./TicTacToe.h"
#include "./NeuralNet.h"
#include <signal.h>

//adding ability to break with ctrl + c
volatile sig_atomic_t stop;
void inthand(int signum){
    stop = 1;
}

#define EPISODES 100
#define GAMES 100 // # of games per episode
#define GAMMA .9 // discount rate of future rewards

/*Things to look at:
 * cache updates (in a batch they update during instead of after like the weights)
 * TODO:
 * Next qmax:
    * Create next Qmax in transition struct
    * rewrite last move's qmax with the qval of the move selected in this move
    * USE THE MOVE SELECTED NOT THE MAX (they are different since we pick available moves, not the best score everytime)
 * Negative reward for losing game
*/

using std::vector;
using std::cout;

//Defines a transition, used in experience replay for learning
struct Transition {
    vector<double> state;
    int action;
    int reward;
    vector<double> nextState;
    vector<double> qvals;
};

namespace QLearning{

    //returns the position of the largest value in the vector
    int argmax(vector<double> x){
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

    //Takes a mini batch of transitions, finds targets then fits to the network
    void network_fit(Network &net, vector<Transition> &miniBatch){
        vector<vector <double>> input, tout;
        vector<double> target;

        for(int i = 0; i < miniBatch.size(); i++){
            Transition t = miniBatch[i];
            input.push_back(t.state);
            
            target = t.qvals;
            vector<double> qvals = net.run(t.nextState);
            target[t.action] = t.reward == 10 ? t.reward : t.reward + GAMMA * qvals[argmax(qvals)]; // terminal state if reward is 10
            tout.push_back(target);
        }

        net.batch_fit(input, tout);
    }


    //erases element in vector if found
    void erase(vector<int> &x, int target){
        for (int i = 0; i < x.size(); i++){
            if (x[i] == target){
                x.erase(x.begin() + i);
                return;
            }
        }
    }


    //picks an action based on the qvalues and possible moves
    // Passing avals by value becuase we are making changes we do not want to keep later
    int pick_action(vector<double> avals, vector<int> &possMoves){
        vector<int> moves = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        
        //removes valid moves from 'moves', leaving only invalid ones
        for(int i = 0; i < possMoves.size(); i++){
            QLearning::erase(moves, possMoves[i]);
        }

        //set aval of invalid moves to -5000 so we do not select an invalid move
        for(int i = 0; i < moves.size(); i++){
            avals[moves[i]] = -5000;
        }

        int action = QLearning::argmax(avals); //find action
        QLearning::erase(possMoves, action); //update possMoves
        return action;
    }


    //Makes a move based on action and retruns the reward
    int make_move(TicTacToe &game, int action){
        int reward;

        // Will always be able to make a move becuase we only pick valid moves
        if(game.makeMove(action + 1)){
            if(game.checkWin())
                reward = 50; // if we win
            else
                reward = 5; // if we make successful move
        }

        return reward;
    }

    
    //Gives a normalized network input for a given game
    // First 9 are position of X's, second 9 are positions of O's, and last 9 are position of blanks 
    vector<double> get_input(TicTacToe &game){
        vector<int> board = game.getBoard();
        vector<double> input(27, 0);
        
        for(int i = 0; i < 9; i++){
            switch(board[i]){
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



    void run(Network &net){
        srand(time(NULL)); // initializes random seed

        //game independent variables
        double eps = 1; // for e-greedy selection (from 1 to .1, then stays at .1)
        int minibatchSize = 16; // mini batch sample size from the replays
        int replaySize = 128;
        vector<Transition> replays, miniBatch(minibatchSize);
        int moves = 0; // total number of moves so far by the network (for replay rewriting)

        //game depedendent variables
        int action, reward;
        vector<double> state, newState, qvals;

        cout << "Episode:\t% NN won:\t% Rand won:\t% Draws:" << endl; // text for displaying % games won each episode
        signal(SIGINT, inthand); // able to break the program with ctrl + c

        for(int i = 0; i < EPISODES; i++){
            int winsNN = 0, winsRand = 0, draws = 0;

            for(int j = 0; j < GAMES; j++){
                TicTacToe game; // Initializes a game
                state = QLearning::get_input(game); // sets the initial state
                vector<int> possibleMoves = {0, 1, 2, 3, 4, 5, 6, 7, 8}; // possible moves for random player

                while(true){ // runs until game is over                
                    //Select action using e-greedy:
                    qvals = net.run(state);
                    if((double)rand() / (double)RAND_MAX < eps){
                        action = possibleMoves[rand() % possibleMoves.size()]; // picks random move from possible moves
                        QLearning::erase(possibleMoves, action); // updates possible moves
                    }
                    else 
                        action = QLearning::pick_action(qvals, possibleMoves);                    

                    //Take action, observe reward and new state:
                    reward = QLearning::make_move(game, action);
                    newState = QLearning::get_input(game);

                    //Store transition in replay:
                    Transition t = {state, action, reward, newState, qvals};
                    if(replays.size() < replaySize)
                        replays.push_back(t);
                    else { // replays is full, so rewrite old ones
                        replays[moves % replaySize] = t;
                    }

                    //Sample mini batch of transitions from replays, then fit to the network:
                    if(replays.size() >= minibatchSize){
                        for (int k = 0; k < minibatchSize; k++)
                            miniBatch[k] = replays[rand() % replays.size()];

                        QLearning::network_fit(net, miniBatch);
                    }
                    
                    if(game.checkWin()){ // check to see if network won
                        winsNN++;
                        break;
                    }
                    else if (possibleMoves.size() == 0){ // draw
                        draws++;
                        break;
                    }

                    //Next players turn (randomly picks place)
                    game.nextTurn();
                    action = possibleMoves[rand() % possibleMoves.size()]; // Player takes turn at random
                    game.makeMove(action + 1);
                    QLearning::erase(possibleMoves, action);
                    if (game.checkWin()){ // check to see if random player won
                        winsRand++;
                        break;
                    }
                    else if (possibleMoves.size() == 0){ // draw
                        draws++;
                        break;
                    }

                    //Update variables for next turn
                    game.nextTurn();
                    moves++;
                    if(eps > .1)
                        eps -= .9 / (.1 * EPISODES * GAMES);
                    
                }

            }
            cout << "\t" << i + 1 << "\t\t" << ((double)winsNN / GAMES) * 100 << "%\t\t" << ((double)winsRand / GAMES) * 100 
                << "%\t\t" << ((double)draws / GAMES) * 100 << "%" << endl;
        }
    }

    void playGame(Network &net){
        TicTacToe game;
        int player = 0;
        vector<int> possibleMoves = {0, 1, 2, 3, 4, 5, 6, 7, 8}; // possible moves

        while(true){
            system("cls");
            game.printBoard(); // show user the current board
            cout << endl;

            //Network's turn
            vector<double> qvals = net.run(QLearning::get_input(game));
            int move = QLearning::pick_action(qvals, possibleMoves) + 1;
            game.makeMove(move);
            if(game.checkWin()){ // if there is a winner then break the game loop, else next move
                player = 1;
                break;
            }
            else
                game.nextTurn();


            system("cls");
            
            game.printBoard(); // show user the current board
            cout << endl;
            for(int i = 0; i < 9; i++){
                cout << qvals[i] << endl;
            }
            //Player's turn
            move = game.getMove(); // get a move 
            while(!game.makeMove(move)){ // validate space is open (make the move if it is)
                cout << "Move Invalid: space is occupied" << endl << endl;
                move = game.getMove();
            }

            QLearning::erase(possibleMoves, move - 1); //update possible moves

            if(game.checkWin()) // if there is a winner then break the game loop, else next move
                break;
            else
                game.nextTurn();
        }

        //prints out winning game board and which player wins
        system("cls");
        game.printBoard();
        cout << endl << endl << "PLAYER " <<
        (player == 1 ? "1 " : "2 ") << "WINS!!" << endl;
    }

}


#endif