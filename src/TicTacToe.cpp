#include "../lib/TicTacToe.h"

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;

TicTacToe::TicTacToe(){
    this->curPlayer = 1;
    
    for(int i = 0; i < 9; i++)
        board.push_back(2);
}

// does not need to do anything as we do not dynamically allocate any memory
TicTacToe::~TicTacToe(){}; 

void TicTacToe::run(){
    while(true){
        system("cls");
        this->printBoard(); // show user the current board
        cout << endl;

        int move = this->getMove(); // get a move 
        while(!this->makeMove(move)){ // validate space is open (make the move if it is)
            cout << "Move Invalid: space is occupied" << endl << endl;
            move = this->getMove();
        }

        if(this->checkWin()) // if there is a winner then break the game loop, else next move
            break;
        else
            this->nextTurn();
    }

    //prints out winning game board and which player wins
    system("cls");
    this->printBoard();
    cout << endl << endl << "PLAYER " <<
    (this->curPlayer == 1 ? "1 " : "2 ") << "WINS!!" << endl;
}

void TicTacToe::printBoard()
{
    // 0 is O's, 1 is X's, and 2 is blank space
    string graphics[3][5] = {
        {"   ___   ", "  /   \\  ", " |     | ", " |     | ", "  \\___/  "},
        {"  \\   /  ", "   \\ /   ", "    X    ", "   / \\   ", "  /   \\  "},
        {"         ", "         ", "         ", "         ", "         "}};

    cout << graphics[this->board[0]][0] << "|" << graphics[this->board[1]][0] << "|" << graphics[this->board[2]][0] << endl
        << graphics[this->board[0]][1] << "|" << graphics[this->board[1]][1] << "|" << graphics[this->board[2]][1] << endl
        << graphics[this->board[0]][2] << "|" << graphics[this->board[1]][2] << "|" << graphics[this->board[2]][2] << endl
        << graphics[this->board[0]][3] << "|" << graphics[this->board[1]][3] << "|" << graphics[this->board[2]][3] << endl
        << graphics[this->board[0]][4] << "|" << graphics[this->board[1]][4] << "|" << graphics[this->board[2]][4] << endl
        << " 1       | 2       | 3       " << endl
        << "---------|---------|---------" << endl
        << graphics[this->board[3]][0] << "|" << graphics[this->board[4]][0] << "|" << graphics[this->board[5]][0] << endl
        << graphics[this->board[3]][1] << "|" << graphics[this->board[4]][1] << "|" << graphics[this->board[5]][1] << endl
        << graphics[this->board[3]][2] << "|" << graphics[this->board[4]][2] << "|" << graphics[this->board[5]][2] << endl
        << graphics[this->board[3]][3] << "|" << graphics[this->board[4]][3] << "|" << graphics[this->board[5]][3] << endl
        << graphics[this->board[3]][4] << "|" << graphics[this->board[4]][4] << "|" << graphics[this->board[5]][4] << endl
        << " 4       | 5       | 6       " << endl
        << "---------|---------|---------" << endl
        << graphics[this->board[6]][0] << "|" << graphics[this->board[7]][0] << "|" << graphics[this->board[8]][0] << endl
        << graphics[this->board[6]][1] << "|" << graphics[this->board[7]][1] << "|" << graphics[this->board[8]][1] << endl
        << graphics[this->board[6]][2] << "|" << graphics[this->board[7]][2] << "|" << graphics[this->board[8]][2] << endl
        << graphics[this->board[6]][3] << "|" << graphics[this->board[7]][3] << "|" << graphics[this->board[8]][3] << endl
        << graphics[this->board[6]][4] << "|" << graphics[this->board[7]][4] << "|" << graphics[this->board[8]][4] << endl
        << " 7       | 8       | 9       " << endl;
}

vector<int> TicTacToe::getBoard() const
{  return this->board;  }

// Makes a move base on the position, returns false if invalid move
//pos is 1-9 position listed on printed board
bool TicTacToe::makeMove(int pos)
{
    if (this->board[pos-1] == 2){
        this->board[pos-1] = this->curPlayer;
        return true;
    }
    else return false;
}

//gets a move from the user
int TicTacToe::getMove()
{
    int move;

    do{
        cout << "Enter position of move you would like to make (int): " << endl;
        cin >> move;
        if(move > 9 || move < 1)
            cout << "Move Invalid: Invalid position." << endl << endl;
    }
    while(move > 9 || move < 1);

    return move;
}

void TicTacToe::nextTurn()
{  ++this->curPlayer %= 2;  }

// checks to see if the game has been won
bool TicTacToe::checkWin(vector<int> board)
{
    //check along rows:
    // if any value in the row does not belong to the current player then 
    // they have not satisfied the win condition across the row
    for(int i = 0; i <= 6; i += 3){
        bool flag = true;

        for(int j = 0; j < 3; j++){
            if((board.empty() ? this->board : board)[i+j] != this->curPlayer)
                flag = false;
        }

        if(flag)
            return true;
    }

    //check along columns:
    // if any value in the column does not belong to the current player then
    // they have not satisfied the win condition across the column
    for(int i = 0; i < 3; i++){
        bool flag = true;

        for(int j = 0; j <= 6; j += 3){
            if((board.empty() ? this->board : board)[i + j] != this->curPlayer)
                flag = false;
        }

        if(flag)
            return true;
    }

    //check diagonals
    // if any value in the diagonal does not belong to the current player then
    // they have not satisfied the win condition across the diagonal
    bool flag = true;
    for(int i = 0; i <= 8; i += 4){
        if((board.empty() ? this->board : board)[i] != this->curPlayer)
            flag = false;
    }
    if(flag)
        return true;

    flag = true;
    for (int i = 2; i <= 6; i += 2){
        if ((board.empty() ? this->board : board)[i] != this->curPlayer)
            flag = false;
    }
    if(flag)
        return true;

    return false; // no win conditions were met
}

// checks to see if the game has ended in a draw
bool TicTacToe::checkDraw(vector<int> board){
    for(int i = 0; i < 9; i++){
        if((board.empty() ? this->board : board)[i] == 2)
            return false;
    }

    return true;
}