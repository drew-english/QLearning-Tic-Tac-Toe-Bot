#include "lib/TicTacToe.h"
#include "lib/NeuralNet.h"
#include "lib/QLearning.h"
#include <signal.h>

using std::cout;

int main(int argc, char *argv[])
{
    srand(time(NULL)); // initializes random seed

    QLearning::run_test();

    return 0;
}