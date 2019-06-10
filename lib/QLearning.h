#ifndef QLEARNING_H
#define QLEARNING_H

#include "./TicTacToe.h"
#include "./NeuralNet.h"
#include "./Player.h"
#include <signal.h>

//adding ability to break with ctrl + c
volatile sig_atomic_t stop;
void inthand(int signum){
    stop = 1;
}

#define EPISODES 100
#define GAMES 100 // # of games per episode

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

namespace QLearning{
    int winsNN = 0, winsRand = 0, draws = 0;

    bool check_end(TicTacToe &game, NNPlayer &netPlayer, bool isNetPlayer){
        if(game.checkWin()){
            if(isNetPlayer){
                netPlayer.final_reward(1); // netplayer wins
                winsNN++;
            }

            if(!isNetPlayer){
                netPlayer.final_reward(-1); // other player wins
                winsRand++;
            }
            return true;
        }
        
        if(game.checkDraw()){
            netPlayer.final_reward(0); // draw
            draws++;
            return true;
        }

        return false;
    }

    void run(NNPlayer netPlayer, MINMAXPlayer mmPlayer){

        cout << "Episode:\t% NN won:\t% Rand won:\t% Draws:" << endl; // text for displaying % games won each episode
        signal(SIGINT, inthand); // able to break the program with ctrl + c

        for(int i = 0; i < EPISODES; i++){
            for(int j = 0; j < GAMES; j++){
                TicTacToe game; // Initializes a game

                while(true){ // runs until game is over                
                    //netplayer's move
                    netPlayer.move(game);                    
                    if(QLearning::check_end(game, netPlayer, true)){
                        break;
                    }
                    else
                        game.nextTurn();
                    
                    //other Player's move
                    mmPlayer.move(game);
                    if(QLearning::check_end(game, netPlayer, false)){
                        break;
                    }
                    else
                        game.nextTurn();
                }

            }
            cout << "\t" << i + 1 << "\t\t" << ((double)winsNN / GAMES) * 100 << "%\t\t" << ((double)winsRand / GAMES) * 100 
                << "%\t\t" << ((double)draws / GAMES) * 100 << "%" << endl;
            winsNN = 0;
            winsRand = 0;
            draws = 0;
        }
    }

    void playGame(NNPlayer netPlayer){
        TicTacToe game;
        int player = 0;
        vector<int> possibleMoves = {0, 1, 2, 3, 4, 5, 6, 7, 8}; // possible moves

        while(true){
            system("cls");
            game.printBoard(); // show user the current board
            cout << endl;

            //Network's turn
            vector<double> qvals = netPlayer.get_qvals(game);
            netPlayer.move(game);
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
            int move = game.getMove(); // get a move 
            while(!game.makeMove(move)){ // validate space is open (make the move if it is)
                cout << "Move Invalid: space is occupied" << endl << endl;
                move = game.getMove();
            }

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