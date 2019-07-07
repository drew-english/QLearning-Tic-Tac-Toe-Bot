#include "lib/TicTacToe.h"
#include "lib/NeuralNet.h"
#include "lib/QLearning.h"
#include <signal.h>

using std::cout;

int main(int argc, char *argv[])
{
    srand(time(NULL)); // initializes random seed

    // loads bestnet and plays a game
    Network net("Saves/BestNet.data");
    NNPlayer netPlayer(&net);
    QLearning::playGame(&netPlayer);

    return 0;
}