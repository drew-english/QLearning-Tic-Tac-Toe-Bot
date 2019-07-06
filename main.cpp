#include "lib/TicTacToe.h"
#include "lib/NeuralNet.h"
#include "lib/QLearning.h"
#include <signal.h>

using std::cout;

int main(int argc, char *argv[])
{
    srand(time(NULL)); // initializes random seed

    Network net(27, 1, 243, 9, relu, linear);
    NNPlayer netPlayer(&net, true);
    MINMAXPlayer mmp;
    
    QLearning::run_test(&mmp, &netPlayer, "testdata/MMP-NET(1).data");
    return 0;
}