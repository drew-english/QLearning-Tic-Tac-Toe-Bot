#include "../lib/Player.h"

//MINMAXPlayer functions:

MINMAXPlayer::MINMAXPlayer(){
    this->valWin = 1;
    this->valDraw = 0;
    this->valLoss = -1;
    this->side = 0;
}


MINMAXPlayer::~MINMAXPlayer(){} // do not need to destruct anything


void MINMAXPlayer::move(TicTacToe &game){
    vector<int> move = max(game); // returns max [score, action] based on game
    game.makeMove(move[1] + 1);
}


vector<int> MINMAXPlayer::max(TicTacToe game){
    vector<vector <int>> bestMoves;
    vector<int> board = game.getBoard();

    // check if game state is already stored (count can only be 1 or 0)
    if(this->moveCache.count(board)){ 
        bestMoves = moveCache[board];
        return bestMoves[rand() % bestMoves.size()];
    }


    //  check if game instance has already finished
    if(game.checkWin(board)){ // check for this player's win
        bestMoves = {{this->valWin, -1}};
        moveCache[board] = bestMoves;
        return bestMoves[0];
    }
    
    game.nextTurn(); // switch to other players turn for win checking
    if(game.checkWin(board)){ // check for other players win
        bestMoves = {{this->valLoss, -1}};
        moveCache[board] = bestMoves;
        return bestMoves[0];
    }
    game.nextTurn(); // switch back to this player's turn


    // building best moves
    int maxVal = this->valDraw;
    int action = -1;
    bestMoves = {{maxVal, action}}; // set to draw value in case there are 0 possible moves
    vector<int> possMoves = get_moves(game);

    for(int i = 0; i < possMoves.size(); i++){
        // create new instance of current game, then execute a possible move
        TicTacToe g = game;
        g.makeMove(possMoves[i] + 1);

        vector<int> res = min(g); // find the result of a min for next player

        if(res[0] > maxVal || action == -1){ //overwrite best moves on finding a larger res or the first possible move
            action = possMoves[i];
            maxVal = res[0];
            bestMoves = {{maxVal, action}};
        }
        else if(res[0] == maxVal){
            action = possMoves[i];
            bestMoves.push_back({maxVal, action});
        }
    }

    moveCache[board] = bestMoves;
    return bestMoves[rand() % bestMoves.size()];
}


vector<int> MINMAXPlayer::min(TicTacToe game){
    vector<vector<int>> bestMoves;
    vector<int> board = game.getBoard();

    // check if game state is already stored (count can only be 1 or 0)
    if (this->moveCache.count(board)){
        bestMoves = moveCache[board];
        return bestMoves[rand() % bestMoves.size()];
    }

    //  check if game instance has already finished
    if (game.checkWin(board)){ // check for this player's win
        bestMoves = {{this->valWin, -1}};
        moveCache[board] = bestMoves;
        return bestMoves[0];
    }

    game.nextTurn(); // switch to other players turn for win checking
    if (game.checkWin(board)){ // check for other players win
        bestMoves = {{this->valLoss, -1}};
        moveCache[board] = bestMoves;
        return bestMoves[0];
    }
    game.nextTurn(); // switch back to this player's turn

    // building best moves
    int maxVal = this->valDraw;
    int action = -1;
    bestMoves = {{maxVal, action}}; // set to draw value in case there are 0 possible moves
    vector<int> possMoves = get_moves(game);

    for (int i = 0; i < possMoves.size(); i++){
        // create new instance of current game, then execute a possible move
        TicTacToe g = game;
        g.nextTurn(); // other player will be making the turn
        g.makeMove(possMoves[i] + 1);
        g.nextTurn(); // resetting to this player's turn

        vector<int> res = max(g); // find the result of a min for next player

        if (res[0] < maxVal || action == -1){ //overwrite best moves on finding a smaller res or the first possible move
            action = possMoves[i];
            maxVal = res[0];
            bestMoves = {{maxVal, action}};
        }
        else if (res[0] == maxVal){
            action = possMoves[i];
            bestMoves.push_back({maxVal, action});
        }
    }

    moveCache[board] = bestMoves;
    return bestMoves[rand() % bestMoves.size()];
}