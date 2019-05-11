#ifndef TICTACTOE_H
#define TICTACTOE_H

#include <vector>
#include <string>
#include <iostream>
#include <stdlib.h>

class TicTacToe {
public:
    TicTacToe(); // constructs a new tic-tac-toe game
    ~TicTacToe(); // destructs a game

    void run(); // runs a new game with 2 players
    void printBoard(); // prints out the board to the console
    std::vector<int> getBoard() const; // returns the state of the game
    void nextTurn(); // iterates to the next player
    bool makeMove(int pos); // makes a new move with current player based on position argument
    int getMove(); // gets a move from the user
    bool checkWin(); // checks for a win with current player
   
private:
    std::vector<int> board; // stores information about the state of the game (1 is X's, 0 is O's, 2 is blank)
    int curPlayer; // stores which players turn it is (1 is player 1, 0 is player 2)
};

#endif