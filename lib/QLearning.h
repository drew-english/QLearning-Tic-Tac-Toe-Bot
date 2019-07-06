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

#define EPISODES 10
#define GAMES 100 // # of games per episode

using std::vector;
using std::cout;

namespace QLearning{
    int winsp1 = 0, winsp2 = 0, draws = 0;

    bool check_end(TicTacToe &game, NNPlayer *netPlayer, bool isNetPlayer, int p){
        if(game.checkWin()){
            p == 1 ? winsp1++ : winsp2++;
            
            if(isNetPlayer && netPlayer != nullptr){
                netPlayer->final_reward(1); // netplayer wins
            }
            if(!isNetPlayer && netPlayer != nullptr){
                netPlayer->final_reward(-1); // other player wins
            }

            return true;
        }
        
        if(game.checkDraw()){
            draws++;

            if(netPlayer != nullptr) 
                netPlayer->final_reward(0); // draw

            return true;
        }

        return false;
    }


    // 2 players play aginst eachother, returns results of training session
    vector<vector<double>> train(Player *p1, Player *p2){
        vector<vector<double>> totalRes; // results of every epoch
        
        cout << "Episode:\t% p1 won:\t% p2 won:\t% Draws:" << endl; // text for displaying % games won each episode
        signal(SIGINT, inthand); // able to break the program with ctrl + c

        // tests which player is the net player ()
        NNPlayer *netPlayer = nullptr;
        if(dynamic_cast<NNPlayer*>(p1) != nullptr)
            netPlayer = (NNPlayer*)p1;
        else if(dynamic_cast<NNPlayer*>(p2) != nullptr)
            netPlayer = (NNPlayer*)p2;

        for(int i = 0; i < EPISODES; i++){
            vector<double> tempRes; // results of this epoch

            for(int j = 0; j < GAMES; j++){
                TicTacToe game; // Initializes a game

                while(true){ // runs until game is over                
                    //Player 1's move
                    p1->move(game);
                    if(QLearning::check_end(game, netPlayer, p1 == netPlayer, 1)){
                        break;
                    }
                    else
                        game.nextTurn();

                    //Player 2's move
                    p2->move(game);                    
                    if(QLearning::check_end(game, netPlayer, p2 == netPlayer, 2)){
                        break;
                    }
                    else
                        game.nextTurn();
                }

            }
            // write results of epoch to the screen
            cout << "\t" << i + 1 << "\t\t" << ((double)winsp1 / GAMES) * 100 << "%\t\t" << ((double)winsp2 / GAMES) * 100 
                << "%\t\t" << ((double)draws / GAMES) * 100 << "%" << endl; 

            // write results of epoch to totalRes
            tempRes.push_back(((double)winsp1 / GAMES) * 100);
            tempRes.push_back(((double)winsp2 / GAMES) * 100);
            tempRes.push_back(((double)draws / GAMES) * 100);
            totalRes.push_back(tempRes);

            // reset wins
            winsp1 = 0;
            winsp2 = 0;
            draws = 0;
        }

        return totalRes;
    }


    //Lets user play against a network player, also displays the qvals the network outputs
    void playGame(NNPlayer netPlayer){
        TicTacToe game;
        int player = 0;
        vector<int> possibleMoves = {0, 1, 2, 3, 4, 5, 6, 7, 8}; // possible moves
        vector<double> qvals = netPlayer.get_qvals(game);

        while(true){
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
            else if(game.checkDraw()){
                player = 2;
                break;
            }
            else
                game.nextTurn();

            //Network's turn
            vector<double> qvals = netPlayer.get_qvals(game);
            netPlayer.move(game);
            
            if(game.checkWin()){ // if there is a winner then break the game loop, else next move
                player = 1;
                break;
            }
            else if(game.checkDraw()){
                player = 2;
                break;
            }
            else
                game.nextTurn();
        }

        //prints out winning game board and which player wins
        system("cls");
        game.printBoard();
        cout << endl << endl;
        switch(player){
            case 0: 
                cout << "PLAYER 1 WINS!!" << endl;
                break;
            case 1:
                cout << "PLAYER 2 WINS!!" << endl;
                break;
            case 2:
                cout << "Draw Game." << endl;
                break;
        }
    }


    // assumes there are 3 different data entries per epoch
    void write_res(string const fileName, vector<vector<double>> &res){
        fstream f(fileName, std::ios::out); // opens a file (creates one if it is not there)

        try {
            if(!f.is_open()) // check to see if open successfully
            throw("File was not opened successfully");
        }
        catch(const char *msg)
        { cerr << "Error while writing results: " << msg << endl;  }

        for(int i = 0; i < res.size(); i++){ // each epoch
            f << res[i][0] << ", " << res[i][1] << ", " << res[i][2] << endl; 
        }
    }


    void run_test(){
        Network net(27, 1, 243, 9, relu, linear);
        NNPlayer netPlayer(&net, true);
        MINMAXPlayer mmp;
        RANDPlayer rp;

        //Train in different scenarios, output to 
        vector<vector<double>> result = QLearning::train(&netPlayer, &rp);
        QLearning::write_res("testdata/NET-RP.data", result);
        
        net = Network(27, 1, 243, 9, relu, linear);
        netPlayer = NNPlayer(&net, true);

        result = QLearning::train(&rp, &netPlayer);
        QLearning::write_res("testdata/RP-NET.data", result);

        net = Network(27, 1, 243, 9, relu, linear);
        netPlayer = NNPlayer(&net, true);

        result = QLearning::train(&netPlayer, &mmp);
        QLearning::write_res("testdata/NET-MMP.data", result);

        net = Network(27, 1, 243, 9, relu, linear);
        netPlayer = NNPlayer(&net, true);

        result = QLearning::train(&mmp, &netPlayer);
        QLearning::write_res("testdata/MMP-NET.data", result);
    }
}


#endif