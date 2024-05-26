#include "framework.h"

int main(int argc, char** argv)
{
	MatrixXf spiralDataX(15, 2);
	spiralDataX << 0.0, 0.0,
		0.22627203, -0.10630602,
		-0.49370526, -0.07908932,
		0.36122907, 0.65727738,
		-0.11046591, -0.99387991,
		0.0, 0.0,
		0.05269733, 0.24438288,
		0.3019796, -0.39850761,
		-0.70998173, 0.2417146,
		0.88005838, -0.4748655,
		0.0, 0.0,
		-0.13811948, -0.20838188,
		-0.02514717, 0.49936722,
		-0.04003289, -0.74893082,
		-0.98680023, -0.16194228;

	MatrixXf spiralDataY(15, 3);
	spiralDataY << 1.0, 0.0, 0.0,
		1.0, 0.0, 0.0,
		1.0, 0.0, 0.0,
		1.0, 0.0, 0.0,
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0;

	Dense dense1(2, 64);
	ReLU activation1;
	Dense dense2(64, 3);
	SoftmaxCategoricalCrossentropyLoss loss_activation;
	OptimizerAdam optimizer(0.02, 1e-5);


	for (int i = 0; i < 10000; ++i)
	{
		// std::cout << "dense1" << "\n";
		dense1.forward(spiralDataX);
		// std::cout << "activation1" << "\n";
		activation1.forward(dense1.outputs);
		// std::cout << "dense2" << "\n";
		dense2.forward(activation1.outputs);
		
		// std::cout << "loss" << "\n";
		float loss = loss_activation.forward(dense2.outputs, spiralDataY);

		std::cout << i << " : Loss -> " << loss << "\n";

		// std::cout << "loss" << "\n";
		loss_activation.backward(loss_activation.outputs, spiralDataY);
		// std::cout << "dense2" << "\n";
		dense2.backward(loss_activation.dinputs);
		// std::cout << "activation1" << "\n";
		activation1.backward(dense2.dinputs);
		// std::cout << "dense1" << "\n";
		dense1.backward(activation1.dinputs);

		// std::cout << "pre update" << "\n";
		optimizer.preUpdateParam();
		// std::cout << "update dense1" << "\n";
		optimizer.updateParams(dense1);
		// std::cout << "update dense2" << "\n";
		optimizer.updateParams(dense2);
		// std::cout << "post update" << "\n";
		optimizer.postUpdateParams();

		// std::cout << "finished" << "\n\n";
	}

	return 0;
}