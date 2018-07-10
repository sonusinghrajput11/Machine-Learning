
public class LogisticRegression {

	public static void main(String[] args) {

		long[][] inputs = new long[100000][20];
		long[] outputs = new long[1000000];
		long[][] tests = {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
				{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},
				{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1},
				{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0},
				{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0}}; //1,4,5,6,10

		int epochs = 100;
		double[] weights = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		double bias = 0.0;
		double learningRate = 0.1;

		for (int i = 0; i < inputs.length; i++) {
			for (int j = 0; j < inputs[i].length; j++) {
				inputs[i][j] = 0;
			}
		}

		for (int i = 0; i < inputs.length; i++) {
			int temp = i;
			int j = inputs[i].length - 1;
			while (temp > 0) {
				int a = temp % 2;
				inputs[i][j] = a;
				temp = temp / 2;
				j--;
			}
			outputs[i] = i % 2 == 0 ? 1 : 0; // even -> 1, odd -> 0
		}

		for (int j = 0; j < epochs; j++) {
			for (int l = 0; l < weights.length; l++) {
				double weightChange = 0.0;
				double biasChange = 0.0;
				for (int k = 0; k < inputs.length; k++) {
					double output = outputs[k];
					double input = inputs[k][l];

					double temp = (weights[l] * input) + bias;
					
					double yPredicted = 1 / (1 + (Math.pow(Math.E, -temp)));
					weightChange = weightChange + ((yPredicted - output) * input);
					biasChange = biasChange + (yPredicted - output);
				}

				weightChange = weightChange / inputs.length;

				biasChange = biasChange / inputs.length;

				weights[l] = weights[l] - (learningRate * weightChange);

				bias = bias - (learningRate * biasChange);

			}
			
			System.out.println("Epoch::" + j + " weight::" + weights + " bias::" + bias);
		}

		System.out.println("Training completed --> weight::" + weights + " bias::" + bias);

		
		for (int l = 0; l < tests.length; l++) {
			double temp = 0.0;
			for(int n = 0; n < tests[l].length; n++){
				temp = temp + (weights[n] * tests[l][n]) + bias;
			}
			double yPredicted = 1 / (1 + (Math.pow(Math.E, -temp)));

			int prediction = yPredicted < 0.5 ? 0 : 1;

			System.out.println(" yyPredicted::" + yPredicted + " Prediction::" + prediction);
		}

	}

}
