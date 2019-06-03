#include "lib/TicTacToe.h"
#include "lib/NeuralNet.h"
#include "lib/QLearning.h"
#include <signal.h>

using std::cout;

int main(int argc, char *argv[])
{
    Network net(27, 2, 108, 9, relu, linear);

    QLearning::run(net);
    net.save("Saves/NetworkSave.data");

    // Network Testing:
    // vector<double> input = {1, 1, 1, 1};
    // vector<double> tout = {5, 5, 5}, out;
    // vector<vector <double>> inputs(8), touts(8);
    // Network testNet(4, 1, 10, 3, relu, linear);

    // out = testNet.run(input);
    // for(int i = 0; i < 3; i++)
    //     cout << out[i] << endl;

    // for(int i = 0; i < 2064; i++){
    //     inputs[i % 8] = input;
    //     touts[i % 8] = tout;
    //     if((i + 1) % 8 == 0)
    //         testNet.batch_fit(inputs, touts);
    // }

    // cout << endl;

    // out = testNet.run(input);
    // for(int i = 0; i < 3; i++)
    //     cout << out[i] << endl;

    // Network net("Saves/NetworkSave.data");

    // QLearning::playGame(net);

    return 0;
}