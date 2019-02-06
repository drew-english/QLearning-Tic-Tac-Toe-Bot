#ifndef TICTACTOE_H
#define TICTACTOE_H

#include <vector>
#include <string>
#include <iostream>
#include <stdlib.h>

using std::vector;
using std::string;
using std::cout;
using std::endl;

class TicTacToe 
{
public:
   TicTacToe(); // constructs a new tic-tac-toe game
   ~TicTacToe(); // destructs a game

   void run(); // runs a new game with 2 players
   void printBoard(); // prints out the board to the console
   vector<int> getBoard() const; // returns the state of the game
   bool makeMove(int pos); // makes a new move based on position argument
   bool checkWin(); // checks for a win with current player
   
private:
   vector<int> board; // stores information about the state of the game
   int curPlayer; // stores which players turn it is
};

#endif