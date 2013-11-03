package edu.cmu.lti.ml.antm.model;

import weka.classifiers.Classifier;
import edu.cmu.lti.ml.antm.data.TestPair;

public class LWLTuner implements ModelTuner {

	private final String classifierName = "weka.classifiers.lazy.LWL";
	private final String[] param = new String[]{"-K", "-U"};
	private final int[] initVal = new int[]{-1, 0};
	private final int[] endVal = new int[]{80, 4};
	private final int[] step = new int[]{3, 1};
	
	public LWLTuner(){
		assert (param.length == initVal.length);
		assert (param.length == endVal.length);
		assert (param.length == step.length);
	};

	public tunedClassifierInfo getTunedModel(TestPair[] testPair) throws Exception {
		
		String trainPath, testPath;
		
		int totalParams = param.length;
		String[] bestOptions = new String[2*totalParams];
		double lowestAvgError = Double.MAX_VALUE;
		
		for(int i=0; i<totalParams; i++){
			int currentOptValue = initVal[i];
			String[] currentOptions = new String[2*(i+1)];
			lowestAvgError = Double.MAX_VALUE;
			
			//get best values for previously computed parameters
			for(int k=0; k<(i+1)*2; k++){
				currentOptions[k] = bestOptions[k];
			}
			currentOptions[2*i] = param[i];
			
			while(currentOptValue <= endVal[i]){
				
				currentOptions[2*i + 1] = String.valueOf(currentOptValue);
				double sumOfError = 0.d;
				
				for(int c=0; c<testPair.length; c++){
					String[] currentOptionsClone = currentOptions.clone();
					trainPath = testPair[c].getTrainFilePath();
					testPath = testPair[c].getTestFilePath();
					sumOfError += WekaModelBuilder.calculateErrorForModel(classifierName, trainPath, testPath, currentOptionsClone);
				}
				double currentAvgError = sumOfError / (double)testPair.length;
				
				for(String s : currentOptions){System.out.print(s+" ");};
				System.out.println(": " + currentAvgError);
				
				if(lowestAvgError > currentAvgError){
					lowestAvgError = currentAvgError;
					bestOptions[2*i] = param[i];
					bestOptions[2*i+1] = String.valueOf(currentOptValue);
				}
				
				currentOptValue += step[i];
			}
		}
		
		System.out.println("Best parameters for LWL: ");
		for(int i=0; i<bestOptions.length; i+=2){
			System.out.println(bestOptions[i]+": "+bestOptions[i+1]);
		}
		
		return new tunedClassifierInfo(this.classifierName, lowestAvgError, bestOptions);
	}



}
