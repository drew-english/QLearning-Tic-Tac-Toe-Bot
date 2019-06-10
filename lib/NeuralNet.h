#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>

using std::vector;
using std::endl;
using std::cerr;
using std::fstream;
using std::string;

//parameters for updating weights (uses RMSProp algorithm)
#define LR .01
#define DECAYRATE .9
#define EPS .000001

class Network {
public:
    // constructs a new network based on parameters
    Network(int inputs, int hiddenLayers, int numHidden,
    int outputs, double(*actHidden)(double x), double(*actOut)(double x));
    Network(char const location[]); // constructs network from a save file
    ~Network(); // destructor for a network

    vector<double> run(vector<double> const &input); //computes the given output of the network from the input
    void fit(vector<double> &input, vector<double> const &target); //Updates the weights of the nework based on the paramters
    void save(char const location[]);
    void load(char const location[]);
    
    //takes multiple inputs and outputs the performs a batch update
    void batch_fit(vector<vector<double>> &input, vector<vector<double>> const &target);

private:
    int inputs, hiddenLayers, numHidden, outputs, totalWeights; // total # of each value
    vector<double> cache, hiddenNeurons, weights; // stores real values of each
    double(*actFunOut)(double x); // neuron activation function for the output
    double(*actFunHidden)(double x); // neuron activation function of the hidden layers

    // gets the weight updates for input and target output then returns them
    void weight_updates(vector<double> &init, vector<double> &input, vector<double> const &target);
};

typedef double (*Function)(double x);

//acivation functions and their derivatives
//relu gives values [0, infinity] while sigmoid gives [0, 1]
//sigmoid is usually better for probabilities while relu is usually better for real values
//linear for range of real valued outputs
double sigmoid(double x);
double d_sigmoid(double x);
double relu(double x);
double d_relu(double x);
double unrelu(double x);
double linear(double x);
double d_linear(double x);

#endif // NerualNet.h
