package edu.cmu.lti.ml.antm.model;

import java.util.Stack;

import edu.cmu.lti.ml.antm.data.TestPair;

public class LogitTuner implements ModelTuner {

	private final String classifierName = "weka.classifiers.meta.LogitBoost";
	private final String[] paramNames=new String[]{"-I", "-F", "-R"};
	private final int[] beginVals=new int[]{10, 0, 1};
	private final int[] endVals=new int[]{100, 10, 5};
	private final int[] steps=new int[]{10,1,1};
	private final String[] defaults=new String[]{"10", "0", "1"};
	private final double[] shrinkageValues=new double[]{0.0,0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0};
	
	public LogitTuner()
	{
		assert(paramNames.length==beginVals.length);
		assert(paramNames.length==endVals.length);
		assert(paramNames.length==steps.length);
	}
	
	/**
	 * Iterate through all the parameters in a greedy manner to find the best parameters
	 */
	public TunedClassifierInfo getTunedModel(TestPair[] testPairs) throws Exception {
		
		double lowestAvgError=Double.MAX_VALUE;
		
		Stack<String> optionList=new Stack<String>();
		
		
		for(int i=0; i<paramNames.length;i++)
		{
			optionList.push(this.paramNames[i]);
			String best=this.defaults[i];
			for(int j=this.beginVals[i]; j<=this.endVals[i]; j+=this.steps[i])
			{
				String paramValue=String.valueOf(j);
				System.out.println(paramNames[i]+"="+paramValue);
				optionList.push(paramValue);
				
				double avgError=this.getAverageError(testPairs, optionList.toArray(new String[optionList.size()]));
				if(avgError<lowestAvgError)
				{
					lowestAvgError=avgError;
					System.out.println(lowestAvgError);
					best=paramValue;
				}
				optionList.pop();
			}
			optionList.push(best);
		}
		
		optionList.push("-H");
		String bestShrink="1.0";
		for(double shrinkValue:shrinkageValues)
		{
			optionList.push(String.valueOf(shrinkValue));
			System.out.println("-H="+shrinkValue);
			double avgError=this.getAverageError(testPairs, optionList.toArray(new String[optionList.size()]));
			if(avgError<lowestAvgError)
			{
				lowestAvgError=avgError;
				System.out.println(lowestAvgError);
				bestShrink=String.valueOf(shrinkValue);
			}
			optionList.pop();
		}
		optionList.push(bestShrink);
			
		return new TunedClassifierInfo(classifierName, lowestAvgError, optionList.toArray(new String[optionList.size()]));
	}

	/**
	 * Get average error across all data sets
	 * @param testPairs
	 * @param options
	 * @return
	 * @throws Exception
	 */
	public double getAverageError(TestPair[] testPairs, String[] options) throws Exception {
		double sum=0D;

		for (TestPair pair : testPairs) {
			sum += WekaModelBuilder.calculateErrorForModel(classifierName,
					pair.getTrainFilePath(), pair.getTestFilePath(), options.clone());
		}

		double average=sum/testPairs.length;		
		return average;
	}
}
