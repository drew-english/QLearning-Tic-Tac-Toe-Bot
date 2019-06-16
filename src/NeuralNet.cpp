#include "../lib/NeuralNet.h"
static vector<double> vcache, mcache; // cache variables for descent optimizer functions
static int numBatches; // nuber of batch updates ran (for adam optimizer)

//Activation funcitons:
double sigmoid(double x)
{ return 1 / (1 + exp(-x)); }

double d_sigmoid(double x)
{ return sigmoid(x) * (1 - sigmoid(x)); }

double relu(double x) // softplus relu
{ return log10(1 + exp(x)); }

double d_relu(double x)
{ return exp(x) / (1 + exp(x)); }

double unrelu(double x)
{	return log(pow(10, x) - 1); }

double linear(double x)
{  return x; }

double d_linear(double x)
{	return 1; }

//RMSProp gradient descent optimizer
double rms_prop(double dx, int index){
    //cache initialized in the network constructor

    vector<double>::iterator c = vcache.begin() + index; // points to the correct spot in cache
    *c = RMSDECAY * (*c) + (1 - RMSDECAY) * pow(dx, 2);
    return -RMSLR * dx / (sqrt(*c) + RMSEPS);
}

double adam(double dx, int index){
    // Cache and numBatches initialzed in network constructor
    
    vector<double>::iterator v = vcache.begin() + index, m = mcache.begin() + index;
    *m = ADAMB1 * (*m) + (1 - ADAMB1) * dx;
    *v = ADAMB2 * (*v) + (1 - ADAMB2) * pow(dx, 2);
    double mHat = *m / (1 - pow(ADAMB1, numBatches));
    double vHat = *v / (1 - pow(ADAMB2, numBatches));
    return -ADAMLR * mHat / (sqrt(vHat) + ADAMEPS);
}

Network::Network(int inputs, int hiddenLayers, int numHidden,
    int outputs, double(*actHidden)(double x), double(*actOut)(double x), double(*optimizer)(double dx, int index)){
    //checks to make sure fucntion arguments are valid
    try {
        if (inputs <= 0)
            throw("Invalid number of inputs to network");
        if (hiddenLayers < 0)
            throw("Invalid number of hidden layers");
        if (hiddenLayers > 0 && numHidden <= 0)
            throw("Invalid number of hidden neurons");
	    if (outputs <= 0)
            throw("Invalid number of outputs");   
    }
    catch(const char * msg)
    { cerr << "Error During Network Construction: " << msg << '\n'; }

    //calculating totalWeights,
    //adding 1 to neuron layers for a bias in each layer
	if (hiddenLayers != 0)
        this->totalWeights = ((inputs + 1) * numHidden) + ((hiddenLayers - 1) * ((numHidden + 1) * numHidden)) +
  	    ((numHidden + 1) * outputs);
	else
  	    this->totalWeights = (inputs + 1) * outputs;

    this->inputs = inputs;
    this->hiddenLayers = hiddenLayers;
    this->numHidden = numHidden;
    this->totalWeights = totalWeights;
    this->outputs = outputs;
    this->actFunHidden = actHidden;
    this->actFunOut = actOut;
    this->optimizer = optimizer;
    numBatches = 0;

    //sets the derivative needed for activation functions of hidden and output
    if (this->actFunHidden == relu)
        this->dHidden = d_relu;
    if (this->actFunHidden == sigmoid)
        this->dHidden = d_sigmoid;
    if (this->actFunHidden == linear)
        this->dHidden = d_linear;
    if (this->actFunOut == linear)
        this->dOut = d_linear;
    if (this->actFunOut == sigmoid)
        this->dOut = d_sigmoid;
    if (this->actFunOut == relu)
        this->dOut = d_relu;

    vcache = vector<double>(this->totalWeights, 0);
    if(optimizer == adam)
        mcache = vector<double>(this->totalWeights, 0); // only need mcache for adam

    for (int i = 0; i < this->totalWeights; i++){ // randomly setting weights
		this->weights.push_back(1 * ((double)rand() / (double)RAND_MAX));
		if (rand() % 2)
			this->weights[i] *= -1;

        if(i < this->numHidden * this->hiddenLayers)
            this->hiddenNeurons.push_back(0);
	}
}

//Reads network parameters from a file locaction and inits optimizer vars
Network::Network(char const location[]){ 
    this->load(location);

    numBatches = 0;
    vcache = vector<double>(this->totalWeights, 0);
    if(optimizer == adam)
        mcache = vector<double>(this->totalWeights, 0); // only need mcache for adam
}

//no memory is dynamically allocated, besides when inserting into the vector containers
// which is then taken care of by their class.
Network::~Network(){};

vector<double> Network::run(vector<double> const &input){
    //makes sure input is the correct size
    try {
        if((int)input.size() != this->inputs)
            throw("Invalid input size");
    }
    catch(const char *msg)
    { cerr << "Error While Running Network: " << msg << endl;  }


    double sum; // stores running sum for each neuron (before activation function)
    vector<double>::iterator w = this->weights.begin();
    vector<double>::iterator neur = this->hiddenNeurons.begin();

    for(int i = 0; i < this->hiddenLayers; i++){ // each hidden layer
        for(int j = 0; j < this->numHidden; j++){ // each neruon in hidden layer
            sum = *w++; // sum starts with bias

            for(int k = 0; k < (i == 0 ? this->inputs : this->numHidden); k++){ // each neruon in previous layer
                sum += *w++ * (i == 0 ? input[k] : this->hiddenNeurons[k]); // sum += current weight multiplied by previous neuron value
            }

            *neur++ = this->actFunHidden(sum); // current neuron value is updated
        }
    }

    vector<double> output; // output layer that is returned by this function
    for(int i = 0; i < this->outputs; i++){ // each output
        sum = *w++; // sum starts with bias

        for(int j = 0; j < (this->hiddenLayers > 0 ? this->numHidden : this->inputs); j++){ // each neuron in previous layer
            sum += *w++ * (this->hiddenLayers > 0 ? //sum += current weight multiplied by previous neuron value
                this->hiddenNeurons[j + ((this->hiddenLayers - 1) * this->numHidden)] : input[j]);
        }

        output.push_back(this->actFunOut(sum)); // current neuron value is updated
    }

  return output;
}

//Calculates deltas for each neuron (to use in weight updates)
vector<double> Network::get_deltas(vector<double> &input, vector<double> const &target, vector<double> const &initDelta){
    // run the network with input parameter to compare to target out
    vector<double> o = this->run(input);

    //making a copy of hidden neuron values for the case of the hidden activation being relu
    //as both the unrelu and relu values of the hidden neurons are needed (easier to do derivative)
    vector<double> h;
    if (this->hiddenLayers > 0 && this->actFunHidden == relu){
        for (int i = 0; i < this->hiddenLayers * this->numHidden; i++)
            h.push_back(unrelu(this->hiddenNeurons[i]));
    }
    else {
        h = hiddenNeurons;
    } // h is still used for calculations
    if (this->actFunOut == relu)
    {
        for (int i = 0; i < this->outputs; i++)
            o[i] = unrelu(o[i]);
    }

    vector<double>::iterator w, d;

    //calculate deltas for output layer
    vector<double> delta(this->hiddenLayers * this->numHidden + this->outputs);
    for (int i = 0; i < this->outputs; i++){
        delta[i + this->hiddenLayers * this->numHidden] = this->dOut(o[i]) * (o[i] - target[i]);
    }

    //calculate deltas for hidden layers if any
    for (int i = this->hiddenLayers; i > 0; i--){
        //finds first weight and delta in next layer
        w = this->weights.begin() + ((this->inputs + 1) * this->numHidden) + ((this->numHidden + 1) * this->numHidden * (i - 1)) + 1;
        d = delta.begin() + (this->numHidden * i);

        for (int j = 0; j < this->numHidden; j++){
            delta[this->numHidden * (i - 1) + j] = 0;

            for (int k = 0; k < (i == this->hiddenLayers ? this->outputs : this->numHidden); k++){
                delta[this->numHidden * (i - 1) + j] += w[(this->numHidden + 1) * k + j] * d[k];
            }

            //doing this calculation now to keep hidden all deltas in the same format
            //(makes it easier when changing weights)
            delta[this->numHidden * (i - 1) + j] *= this->dHidden(h[this->numHidden * (i - 1) + j]);
        }
    }

    if(!initDelta.empty()){
        for(int i = 0; i < delta.size(); i++) //accumulating deltas for batch updating
            delta[i] += initDelta[i];
    }

    return delta;
}

//init must be size of totalWeights, weight updates will be added to this vector
// Must have deltas before updating weights
void Network::weight_updates(vector<double> &input, vector<double> const &delta){
    vector<double>::iterator w, n;
    int index;
    
    //delta is set up so that hidden layer neurons comes first,
    //then any other hidden layer's neurons, then output neurons last

    /* updating weights to output layer */

    //first weight (starting with the bias) to the first delta in output layer
    index = (this->hiddenLayers ? (this->inputs + 1) * this->numHidden + 
        (this->numHidden + 1) * this->numHidden * (this->hiddenLayers - 1) : 0);
    w = this->weights.begin() + index;

    //first neuron in the previous layer
    n = (this->hiddenLayers ? this->hiddenNeurons.begin() + (this->numHidden * (this->hiddenLayers - 1)) : input.begin());

    double dx = 0.0;
    for (int i = 0; i < this->outputs; i++){
        for (int j = 0; j < (this->hiddenLayers ? this->numHidden : this->inputs) + 1; j++){
            if (j == 0){
                dx = delta[this->numHidden * this->hiddenLayers + i];
                *w++ += this->optimizer(dx, index++);
            }
            else{
                dx = delta[this->numHidden * this->hiddenLayers + i] * n[j - 1];
                *w++ += this->optimizer(dx, index++);
            }
        }
    }

    /* updating weights to hidden layers if any */

    for (int i = this->hiddenLayers; i > 0; i--){
        //first weight (and cache) (starting with bias) to the delta in current layer
        index = ((i == 1 ? 0 : 1) * (this->inputs + 1) * this->numHidden) +
            ((i == 1 ? 0 : i - 2) * (this->numHidden + 1) * this->numHidden);
        w = this->weights.begin() + index;
                
        //first neuron in previous layer
        n = (i == 1 ? input.begin() : this->hiddenNeurons.begin() + (this->numHidden * (i - 2)));

        for (int j = 0; j < this->numHidden; j++){
            for (int k = 0; k < (i == 1 ? this->inputs : this->numHidden) + 1; k++){
                if (k == 0){
                    dx = delta[this->numHidden * (i - 1) + j];
                    *w++ += this->optimizer(dx, index++);
                }
                else{
                    dx = delta[this->numHidden * (i - 1) + j] * n[k - 1];
                    *w++ += this->optimizer(dx, index++);
                }
            }
        }
    }
}   


void Network::fit(vector<double> &input, vector<double> const &target){
    vector<double> delta = get_deltas(input, target);
    weight_updates(input, delta); // weight_updates directly change the weights
}

void Network::batch_fit(vector<vector<double>> &input, vector<vector<double>> const &target){
    vector<double> deltas(this->hiddenLayers * this->numHidden + this->outputs, 0);
    vector<double> avgIn(input[0].size(), 0);
    
    numBatches++; // update number increment for adam 
    for(int i = 0; i < input.size(); i++){
        deltas = get_deltas(input[i], target[i], deltas); // summation of deltas from all inputs and targets
        for(int j = 0; j < input[i].size(); j++)
            avgIn[j] += input[i][j];
    }

    for(int i = 0; i < deltas.size(); i++){
        deltas[i] /= input.size();  // average of each delta
        if(i < avgIn.size())
            avgIn[i] /= input.size(); // average input
    }

    weight_updates(avgIn, deltas); // input and target aren't used

    //reset cache after each batch
    // this->cache.clear();
    // this->cache = vector<double> (this->totalWeights, 0);
}

void Network::save(char const location[])
{
    int funHidden = 0, funOut = 0, opt = 0;

    //establish file stream, error check, and print network information
    fstream out(location);
    try {
        if(!out.is_open())
        throw("File was not opened successfully");
    }
    catch(const char *msg)
    { cerr << "Error while saving network: " << msg << endl;  }

    out << this->inputs << " " << this->hiddenLayers << " " <<
        this->numHidden << " " << this->outputs<< endl;

    if (this->actFunHidden == sigmoid)
        funHidden = 0;
    if (this->actFunHidden == relu)
        funHidden = 1;
    if (this->actFunHidden == linear)
        funHidden = 2;
    if (this->actFunOut == sigmoid)
        funOut = 0;
    if (this->actFunOut == relu)
        funOut = 1;
    if (this->actFunOut == linear)
        funOut = 2;
    if (this->optimizer == adam)
        opt = 0;
    if (this->optimizer == rms_prop)
        opt = 1;

    out << funHidden << " " << funOut << " " << optimizer << endl
        << this->totalWeights << endl;

    for(int i = 0; i < this->totalWeights; i++)
        out << this->weights[i] << endl;

    out.close();
}

void Network::load(char const location[])
{
 
    fstream in(location); // open file
    try { // check for errors
        if (!in.is_open())
        throw("File was not opened successfully");
    }
    catch (const char *msg)
    { cerr << "Error while loading network: " << msg << endl; }

    ActFun actFun[3] = {sigmoid, relu, linear};
    OptFun optFun[2] = {adam, rms_prop};
    int funHidden, funOut, opt;

    //read in all network parameters
    in >> this->inputs >> this->hiddenLayers >> this->numHidden
        >> this->outputs >> funHidden >> funOut >> opt >> this->totalWeights;
    this->actFunHidden = actFun[funHidden];
    this->actFunOut = actFun[funOut];
    this->optimizer = optFun[opt];

    //sets the derivative needed for activation functions of hidden and output
    if (this->actFunHidden == relu)
        this->dHidden = d_relu;
    if (this->actFunHidden == sigmoid)
        this->dHidden = d_sigmoid;
    if (this->actFunHidden == linear)
        this->dHidden = d_linear;
    if (this->actFunOut == linear)
        this->dOut = d_linear;
    if (this->actFunOut == sigmoid)
        this->dOut = d_sigmoid;
    if (this->actFunOut == relu)
        this->dOut = d_relu;

    //clear current values of each vector, then repopulate them
    this->weights.clear();
    vcache.clear();
    mcache.clear();
    this->hiddenNeurons.clear();

    this->hiddenNeurons.resize(this->hiddenLayers * this->numHidden, 0);
    
    double temp;
    for(int i = 0; i < this->totalWeights; i++){
        in >> temp;
        this->weights.push_back(temp);
    }

    in.close();
}
