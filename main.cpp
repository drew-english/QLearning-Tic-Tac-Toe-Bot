#include "lib/TicTacToe.h"
#include "lib/NeuralNet.h"
#include "lib/QLearning.h"
#include <signal.h>

using std::cout;

int main(int argc, char *argv[])
{
    Network net(27, 1, 243, 9, relu, linear);

    QLearning::run(net);
    net.save("Saves/NetworkSave.data");

    return 0;
}