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

//RMSProp hyperparameters
#define RMSLR .01
#define RMSDECAY .9
#define RMSEPS .000001

//Adam hyperparameters
#define ADAMLR .01
#define ADAMB1 .9
#define ADAMB2 .999
#define ADAMEPS .0000001

//gradient decent optimizers
double rms_prop(double dx, int index);
double adam(double dx, int index);

class Network {
public:
    // constructs a new network based on parameters
    Network(int inputs, int hiddenLayers, int numHidden,
    int outputs, double(*actHidden)(double x), double(*actOut)(double x), double(*optimizer)(double dx, int index) = adam);
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
    vector<double> hiddenNeurons, weights; // stores real values of each
    double(*actFunOut)(double x); // neuron activation function for the output
    double(*dOut)(double x); // derivative of the output activation function
    double(*actFunHidden)(double x); // neuron activation function of the hidden layers
    double(*dHidden)(double x); // derivative of the hidden neuron activation function
    double(*optimizer)(double dx, int index); // gradient decent optimizer for weight updates


    // gets the weight updates for input and target output then returns them
    void weight_updates(vector<double> &input, vector<double> const &deltas = {});

    //Calculates deltas for each neuron (to use in weight updates)
    vector<double> get_deltas(vector<double> &input, vector<double> const &target, vector<double> const &initDelta = {});
};

typedef double (*ActFun)(double x); // activation function type
typedef double (*OptFun)(double dx, int index); // optimizer function type

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

#endif
