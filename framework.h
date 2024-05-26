/**
* @author Chen-Kai Tsai
* @brief beta release
* @version 0.8
* @date 2023-10-30
* 
* @version 0.9
* @brief optimizer bug fix
* @data 2023-11-02
* 
* @todo Private parameter
* @todo check memory usage
*/

#pragma once

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#define __DEBUG__

using Eigen::MatrixXf;
using Eigen::VectorXf;

class Layer
{
protected:
public:
	MatrixXf weights;
	VectorXf biases;
	MatrixXf outputs;

	MatrixXf inputs;

	MatrixXf dinputs;
	MatrixXf dweights;
	VectorXf dbiases;

	// for SGD
	MatrixXf weightMomentums;
	VectorXf biasMomentums;

	// for AdaGrad
	MatrixXf weightCache;
	VectorXf biasCache;

	bool optimizerInit;

	Layer() {}
	~Layer() {}

	virtual void forward(MatrixXf inputs) = 0;
	virtual void backward(MatrixXf dvalues) = 0;
};

class Dense : public Layer
{
public:
	Dense(unsigned int numInputs, unsigned int numNeurons)
	{
		this->weights = 0.01 * MatrixXf::Random(numInputs, numNeurons);
		this->biases = VectorXf::Zero(numNeurons);
		this->optimizerInit = true;
	}
	~Dense() {}

	void forward(MatrixXf inputs)
	{
		this->inputs = inputs;
		this->outputs = inputs * weights;
		this->outputs.rowwise() += biases.transpose();
	}

	void backward(MatrixXf dvalues)
	{
		this->dinputs = dvalues * this->weights.transpose();
		this->dweights = this->inputs.transpose() * dvalues;
		this->dbiases = dvalues.colwise().sum();
	}
};

class ReLU : public Layer
{
public:
	void forward(MatrixXf inputs)
	{
		this->inputs = inputs;
		// Keeping the size of the Matrix
		this->outputs = inputs.cwiseMax(0.0);
	}

	void backward(MatrixXf dvalues)
	{
		this->dinputs = dvalues;
		(this->inputs.array() <= 0).select(0, dinputs.array());
	}
};

class Softmax : public Layer
{
public:
	void forward(MatrixXf inputs)
	{
		this->inputs = inputs;

		inputs = (inputs.array() - inputs.maxCoeff());
		inputs = inputs.array().exp();

		Eigen::VectorXf sum = inputs.rowwise().sum();

		this->outputs = inputs.array().colwise() / sum.array();
	}

	void backward(MatrixXf dvalues)
	{
		// Implemenatation pending
		exit(EXIT_FAILURE);
	}
};

class Loss
{
protected:
	MatrixXf dinputs;

public:
	virtual MatrixXf forward(MatrixXf output, MatrixXf groundTruth) = 0;

	float calculate(MatrixXf output, MatrixXf groundTruth)
	{
		MatrixXf sampleLosses = this->forward(output, groundTruth);
		
		return sampleLosses.sum() / (sampleLosses.cols() * sampleLosses.rows());
	}
};

class CategoricalCrossentropyLoss : public Loss
{
public:
	// Only support one-hot encoding
	MatrixXf forward(MatrixXf output, MatrixXf groundTruth)
	{
		output = output.cwiseMin(1 - 1e-7).cwiseMax(1e-7);
		output = output.cwiseProduct(groundTruth);

		MatrixXf categoricalCrossEntropyLoss = output.rowwise().sum();
		categoricalCrossEntropyLoss.array() = categoricalCrossEntropyLoss.array().log() * -1;

		return categoricalCrossEntropyLoss;
	}

	void backward(MatrixXf dvalues, MatrixXf groundTruth)
	{
		int samples = dvalues.rows();

		this->dinputs = -groundTruth.array() / dvalues.array();
		this->dinputs = this->dinputs.array() / samples;
	}
};

class SoftmaxCategoricalCrossentropyLoss
{
public:

	MatrixXf outputs;
	MatrixXf dinputs;

	SoftmaxCategoricalCrossentropyLoss() {}
	~SoftmaxCategoricalCrossentropyLoss() {}

	float forward(MatrixXf inputs, MatrixXf groundTruth)
	{
		// Don't need to keep these in the class
		// TODO : Move implementation rather than create class.
		Softmax activation;
		CategoricalCrossentropyLoss lossFunc;

		activation.forward(inputs);
		this->outputs = activation.outputs;
		return lossFunc.calculate(this->outputs, groundTruth);
	}

	void backward(MatrixXf dvalues, MatrixXf groundTruth)
	{
		int samples = dvalues.rows();

		// Book sugestion might not be faster
		this->dinputs = dvalues;
		this->dinputs.array() = this->dinputs.array() - groundTruth.array();
		this->dinputs.array() = this->dinputs / static_cast<float>(samples);
	}
};

class OptimizerSGD
{
private:
	float learningRate;
	float curLearningRate;
	float decay;
	float iterations;
	float momentum;
public:
	OptimizerSGD(float learningRate = 1.0, float decay = 0.0, float momentum = 0.0)
	{
		this->learningRate = learningRate;
		this->curLearningRate = learningRate;
		this->decay = decay;
		this->iterations = 0;
		this->momentum = momentum;
	}

	void preUpdateParams()
	{
		// if decay is set, it will always greater than 0
		if (this->decay > 0.0) {
			curLearningRate = learningRate * (1.0 / (1.0 + this->decay * this->iterations));
		}
	}

	void updateParams(Layer& layer)
	{
		// initialize weightMomentums with zeros if not initialized
		if (layer.optimizerInit)
		{
			layer.weightMomentums = MatrixXf::Zero(layer.weights.rows(), layer.weights.cols());
			layer.biasMomentums = VectorXf::Zero(layer.biases.rows(), layer.biases.cols());
			layer.optimizerInit = false;
		}

		// if momentum is set, it will always greater than 0
		if (this->momentum > 0.0)
		{
			MatrixXf weightUpdates = (layer.weightMomentums.array() * this->momentum) - (this->curLearningRate * layer.dweights.array());
			layer.weightMomentums = weightUpdates;

			VectorXf biasUpdates = (layer.biasMomentums.array() * this->momentum) - (this->curLearningRate * layer.dbiases.array());
			layer.biasMomentums = biasUpdates;

			// Update Weights and Biases
			layer.weights += weightUpdates;
			layer.biases += biasUpdates;
		}
		// vanilla SGD Updates
		else
		{
			MatrixXf weightUpdates = layer.dweights.array() * (-1 * this->curLearningRate);
			VectorXf biasUpdates = layer.dbiases.array() * (-1 * this->curLearningRate);

			// Update Weights and Biases
			layer.weights += weightUpdates;
			layer.biases += biasUpdates;
		}
	}

	void postUpateParams()
	{
		this->iterations += 1;
	}
};

class OptimizerAdagrad
{
private:
	float learningRate;
	float curLearningRate;
	float decay;
	float iterations;
	float epsilon;
public:
	OptimizerAdagrad(float learningRate = 1.0, float decay = 0.0, float epsilon = 1e-6)
	{
		this->learningRate = learningRate;
		this->curLearningRate = learningRate;
		this->decay = decay;
		this->iterations = 0;
		this->epsilon = epsilon;
	}

	void updateParams(Layer& layer)
	{
		// initialize weightMomentums with zeros if not initialized
		if (layer.optimizerInit)
		{
			layer.weightCache = MatrixXf::Zero(layer.weights.rows(), layer.weights.cols());
			layer.biasCache = VectorXf::Zero(layer.biases.rows(), layer.biases.cols());
			layer.optimizerInit = false;
		}

		layer.weightCache.array() += (layer.dweights.array().pow(2));
		layer.biasCache.array() += (layer.dbiases.array().pow(2));

		layer.weights.array() += layer.dweights.array() * (-1 * this->curLearningRate) / (layer.weightCache.array().sqrt() + this->epsilon);
		layer.biases.array() += layer.dbiases.array() * (-1 * this->curLearningRate) / (layer.biasCache.array().sqrt() + this->epsilon);
	}

	void postUpdateParams()
	{
		this->iterations += 1;
	}
};

class OptimizerAdam
{
private:
	float learningRate;
	float curLearningRate;
	float decay;
	float iterations;
	float epsilon;
	float beta_1;
	float beta_2;
public:
	OptimizerAdam(float learningRate = 0.001, float decay = 0.0, float epsilon = 1e-6, float beta_1 = 0.9, float beta_2 = 0.999)
	{
		this->learningRate = learningRate;
		this->curLearningRate = learningRate;
		this->decay = decay;
		this->iterations = 0;
		this->epsilon = epsilon;
		this->beta_1 = beta_1;
		this->beta_2 = beta_2;
	}

	void preUpdateParam()
	{
		if (this->decay > 0.0) {
			curLearningRate = learningRate * (1.0 / (1.0 + this->decay * this->iterations));
		}
	}

	void updateParams(Layer& layer)
	{
		if (layer.optimizerInit)
		{
			layer.weightMomentums = MatrixXf::Zero(layer.weights.rows(), layer.weights.cols());
			layer.weightCache = MatrixXf::Zero(layer.weights.rows(), layer.weights.cols());
			layer.biasMomentums = MatrixXf::Zero(layer.biases.rows(), layer.biases.cols());
			layer.biasCache = VectorXf::Zero(layer.biases.rows(), layer.biases.cols());
			layer.optimizerInit = false;
		}

		// Update momentums
		layer.weightMomentums.array() = (layer.weightMomentums.array() * this->beta_1) + (layer.dweights.array() * (1.0 - this->beta_1));
		layer.biasMomentums.array() = (layer.biasMomentums.array() * this->beta_1) + (layer.dbiases.array() * (1.0 - this->beta_1));

		// get corrected momentums
		MatrixXf weightMomentumsCorrected = layer.weightMomentums.array() / (1.0 - std::pow(this->beta_1, (this->iterations + 1)));
		MatrixXf biasMomentumsCorrected = layer.biasMomentums.array() / (1.0 - std::pow(this->beta_1, (this->iterations + 1)));
		
		// Calculate cache
		layer.weightCache.array() = (layer.weightCache.array() * this->beta_2) + (layer.dweights.array().pow(2) * (1.0 - this->beta_2));
		layer.biasCache.array() = (layer.biasCache.array() * this->beta_2) + (layer.dbiases.array().pow(2) * (1.0 - this->beta_2));

		// get corrected cache 
		// TODO : the constant can be calculate and stored for reuse
		MatrixXf weightCacheCorrected = layer.weightCache.array() / (1.0 - std::pow(this->beta_2, (this->iterations + 1)));
		MatrixXf biasCacheCorrected = layer.biasCache.array() / (1.0 - std::pow(this->beta_2, (this->iterations + 1)));


		// Update weights and bias
		layer.weights.array() += weightMomentumsCorrected.array() * (-1 * this->curLearningRate) / (weightCacheCorrected.array().sqrt() + this->epsilon);
		layer.biases.array() += biasMomentumsCorrected.array() * (-1 * this->curLearningRate) / (biasCacheCorrected.array().sqrt() + this->epsilon);
	}

	void postUpdateParams()
	{
		this->iterations += 1;
	}
};
