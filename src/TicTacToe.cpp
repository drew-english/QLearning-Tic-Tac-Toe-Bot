#include "TicTacToe.h"

TicTacToe::TicTacToe()
{
   this->curPlayer = 1;
   
   for(int i = 0; i < 9; i++)
      board.push_back(-1);
}

// does not need to do anything as we do not dynamically allocate any memory
TicTacToe::~TicTacToe(){}; 

void TicTacToe::printBoard()
{
   string graphics[3][6] = {{}};

   system("cls");
   cout << "         |         |         " << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "---------|---------|---------" << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "---------|---------|---------" << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "         |         |         " << endl
        << "         |         |         " << endl;
}