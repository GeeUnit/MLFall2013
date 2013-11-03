package edu.cmu.lti.ml.antm.model;

import java.net.URL;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import edu.cmu.lti.ml.antm.data.TestPair;

public class RFTuner implements ModelTuner {
	private final String classifierName = "weka.classifiers.trees.RandomForest";
	//values for -K will be treated as the % of numAttributes (1->10%, 2->20%, ...)
	private final String[] param = new String[]{"-I", "-K", "-depth"};
	private final int[] initVal = new int[]{50, 1, 3};
	private final int[] endVal = new int[]{150, 10, 30};
	private final int[] step = new int[]{5, 1, 2};
	
	
	public RFTuner(){
		assert (param.length == initVal.length);
		assert (param.length == endVal.length);
		assert (param.length == step.length);
	};

	public Classifier getTunedModel(TestPair[] testPair) throws Exception {
		
		String trainPath, testPath;
		
		int totalParams = param.length;
		String[] bestOptions = new String[2*totalParams];
		
		for(int i=0; i<totalParams; i++){
			double lowestAvgError = Double.MAX_VALUE;
			int currentOptValue = initVal[i];
			String[] currentOptions = new String[2*(i+1)];
			for(int k=0; k<(i+1)*2; k++){
				currentOptions[k] = bestOptions[k];
			}
			currentOptions[2*i] = param[i];
			
			while(currentOptValue <= endVal[i]){
				currentOptions[2*i + 1] = String.valueOf(currentOptValue);
				double currentAvgError = 0.d;
				for(int c=0; c<testPair.length; c++){
					String[] currentOptionsClone = currentOptions.clone();
					if(i>=1){ // is using -K
						//get number of attributes
						ArffLoader testLoader=new ArffLoader();
						URL url=WekaModelBuilder.class.getClassLoader().getResource(testPair[c].getTestFilePath());
						testLoader.setURL(url.toString());
						Instances testInstances=testLoader.getStructure();
						int adjustedValue = (int)(Double.parseDouble(currentOptionsClone[3]) * (0.1d * (double)testInstances.numAttributes()));
						currentOptionsClone[3] = String.valueOf(adjustedValue); 
					}
					trainPath = testPair[c].getTrainFilePath();
					testPath = testPair[c].getTestFilePath();
					currentAvgError += WekaModelBuilder.calculateErrorForModel(classifierName, trainPath, testPath, currentOptionsClone);
				}
				currentAvgError /= testPair.length;
				
				//System.out.print("currentOptions: ");
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
		
		System.out.println("Best parameters for Random Forest: ");
		for(int i=0; i<bestOptions.length; i+=2){
			System.out.println(bestOptions[i]+": "+bestOptions[i+1]);
		}
		
		return Classifier.forName(classifierName, bestOptions);
	}

}
