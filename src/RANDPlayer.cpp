#include "../lib/Player.h"

//RANDPlayer functions:

RANDPlayer::RANDPlayer(){} // do not need to construct anything
RANDPlayer::~RANDPlayer(){} // do not need to destruct anything

void RANDPlayer::move(TicTacToe &game){    
    vector<int> possibleMoves = get_moves(game);
    game.makeMove(possibleMoves[rand() % possibleMoves.size()] + 1); // makes a random move based on possible moves
}
