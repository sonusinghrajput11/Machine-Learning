
public class LogisticRegression {

	public static void main(String[] args) {

		long[][] inputs = new long[100000][17];
		long[] outputs = new long[100000];
		long[][] tests = {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
				{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},
				{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1},
				{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0},
				{0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0}}; //1,4,5,6,10

		int epochs = 1000;
		double[] weights = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		double bias = 0.0;
		double learningRate = 0.5;

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

		for (int i = 1; i <= epochs; i++) {
			double totalLoss = 0.0;
			for (int j = 0; j < weights.length; j++) {
				double weightChange = 0.0;
				double biasChange = 0.0;
				for (int k = 0; k < inputs.length; k++) {
					double output = outputs[k];
					double input = inputs[k][j];
					double temp = 0.0;
					for(int l = 0; l < inputs[k].length; l++){
						double tempInput = inputs[k][l];
						temp = (weights[l] * tempInput) + bias; // 1. Calculate Y' i.e. Y' = w1.x1 + w2.x2 + w3.x3......
					}	
					double yPredicted = 1 / (1 + (Math.pow(Math.E, -temp)));
					double loss = yPredicted - output;              //2. Calculate diff i.e Y' - Y
					
					totalLoss = totalLoss + Math.pow(loss, 2);      //Calculating total loss
					
					weightChange = weightChange + ((loss) * input); //3. Calculating total diff i.e. (Y'1 - Y1)x1 + (Y'2 - Y2)x1 + (Y'3 - Y3)x1.... 
					biasChange = biasChange + (yPredicted - output);
				}

				weightChange = weightChange / inputs.length;		//4. Divide by m

				biasChange = biasChange / inputs.length;

				weights[j] = weights[j] - (learningRate * weightChange); //5. Updating weight i.e. w1 = w1 - alpha*((Y'1 - Y1)x1 + (Y'2 - Y2)x1 + (Y'3 - Y3)x1)

				bias = bias - (learningRate * biasChange);			//6. Updating bias

			}
			
			System.out.println("Epoch::" + i + " Total loss::" + totalLoss + " weight::" + printWeights(weights) + " bias::" + bias);
		}

		System.out.println("Training completed --> weight::" + printWeights(weights) + " bias::" + bias);

		
		for (int l = 0; l < tests.length; l++) {
			double temp = 0.0;
			double input = 0.0;
			for(int n = 0; n < tests[l].length; n++){
				input = input + (Math.pow(2, n) * tests[l][(tests[l].length - 1) - n]);
				temp = temp + (weights[n] * tests[l][n]) + bias;
			}
			double yPredicted = 1 / (1 + (Math.pow(Math.E, -temp)));

			String prediction = yPredicted < 0.5 ? "Odd" : "Even";

			System.out.println("input:::"+ input +" yyPredicted::" + yPredicted + " Prediction::" + prediction);
		}

	}
	
	public static String printWeights(double[] weights){
		String weight = "";
		for(int i = 0; i < weights.length; i++){
			weight = weight + weights[i] + ",";
		}
		
		return weight;
	}

}
