package edu.cmu.lti.ml.antm.model;


import java.io.IOException;
import java.net.URL;
import java.util.PriorityQueue;

import edu.cmu.lti.ml.antm.data.TestPair;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;


public class WekaModelBuilder {
	
	private static String BASELINE_CLASSIFIER="weka.classifiers.bayes.NaiveBayes";

	
	public static double getBestModelError(String[] modelClassNames, TestPair dataSet) throws Exception
	{
		//Classifier baselineModel=trainModel(BASELINE_CLASSIFIER, dataSet.getTrainFilePath());

		double baselineError=calculateErrorForModel(BASELINE_CLASSIFIER, dataSet.getTrainFilePath(), dataSet.getTestFilePath());
		System.out.println("be: " + baselineError);
		
		//minimize error ratio
		double bestErrorRatio=Double.MAX_VALUE;
		Classifier bestModel=null;
		
		for(String classifierName:modelClassNames)
		{
			/*
			Classifier model=trainModel(classifierName, dataSet.getTrainFilePath());
			double accuracy=testModel(model, dataSet.getTestFilePath());
			double errorRatio=(1-accuracy)/(1-baselineAccuracy);
			*/
			
			double errorRatio = calculateErrorForModel(classifierName, dataSet.getTrainFilePath(), dataSet.getTestFilePath())/baselineError;
			
			if(errorRatio<bestErrorRatio)
			{
				Classifier model=trainModel(classifierName, dataSet.getTrainFilePath());
				bestErrorRatio=errorRatio;
				bestModel=model;
			}	
			System.out.println(classifierName+"\terror ratio: "+errorRatio);
		}
		
		System.out.println("Best model was: "+bestModel.getClass().getName()+" with error: "+bestErrorRatio);
		outputModel(bestModel, dataSet.getDescription()+".model");
		
		return bestErrorRatio;
	}
	
	/**
	 * Train a classifier on a specific training set
	 * @param modelClassName
	 * @param trainSet
	 * @return
	 * @throws Exception
	 */
	public static Classifier trainModel(String modelClassName, String trainSet) throws Exception
	{
		URL url=WekaModelBuilder.class.getClassLoader().getResource(trainSet);
		DataSource source = new DataSource(url.getFile());
		Classifier model=Classifier.forName(modelClassName, null);
		
		Instances structure=source.getDataSet();
		structure.setClassIndex(structure.numAttributes()-1);
		
		model.buildClassifier(structure);
		
		return model;
	}
	
	/**
	 * Test a classifier on a specific test set. Returns accuracy.
	 * @param model
	 * @param testSet
	 * @return
	 * @throws IOException
	 * @throws Exception
	 */
	public static double testModel(Classifier model, String testSet) throws IOException, Exception
	{
		ArffLoader testLoader=new ArffLoader();
		URL url=WekaModelBuilder.class.getClassLoader().getResource(testSet);
		testLoader.setURL(url.toString());
		
		Instances testInstances=testLoader.getStructure();
		testInstances.setClassIndex(testInstances.numAttributes()-1);
		
		Instance current;
			
		int correct=0;
		int count=0;
		while((current=testLoader.getNextInstance(testInstances))!=null)
		{
			int AttIdx = testInstances.numAttributes()-1;
			if(testInstances.attribute(AttIdx).value((int)model.classifyInstance(current)).equals(current.stringValue(AttIdx)))
			{
				correct++;
			}
			count++;
		}
		
		double accuracy=correct/(double)count;
		return accuracy;
	}

	
	/*
	 * Also computes error, but now using errorRate()
	 * 
	 */
	public static double calculateErrorForModel(String classifierName, TestPair pair, String[] Options, boolean outputModel) throws Exception
    {
		Classifier model=Classifier.forName(classifierName, Options);
        
        String testPath = pair.getTestFilePath();
        String trainPath = pair.getTrainFilePath();
		
        URL url=WekaModelBuilder.class.getClassLoader().getResource(trainPath);
        DataSource trainSource = new DataSource(url.getFile());

        Instances trainInstances=trainSource.getDataSet();
        trainInstances.setClassIndex(trainInstances.numAttributes()-1);
       
        model.buildClassifier(trainInstances);

        url=WekaModelBuilder.class.getClassLoader().getResource(testPath);
        DataSource testSource = new DataSource(url.getFile());

        Instances testInstances=testSource.getDataSet();
        testInstances.setClassIndex(testInstances.numAttributes()-1);
       
       
        Evaluation eval=new Evaluation(trainInstances);
        eval.evaluateModel(model, testInstances);
        
        if(outputModel){
        	outputModel(model, pair.getDescription()+".model");
        }

        return eval.errorRate();
		
    }
	
	public static double calculateErrorForModel(String classifierName, String trainPath, String testPath, String[] Options) throws Exception
    {
           Classifier model=Classifier.forName(classifierName, Options);
           
          
           URL url=WekaModelBuilder.class.getClassLoader().getResource(trainPath);
           DataSource trainSource = new DataSource(url.getFile());
   
           Instances trainInstances=trainSource.getDataSet();
           trainInstances.setClassIndex(trainInstances.numAttributes()-1);
          
           model.buildClassifier(trainInstances);
   
           url=WekaModelBuilder.class.getClassLoader().getResource(testPath);
           DataSource testSource = new DataSource(url.getFile());
   
           Instances testInstances=testSource.getDataSet();
           testInstances.setClassIndex(testInstances.numAttributes()-1);
          
          
           Evaluation eval=new Evaluation(trainInstances);
          
           eval.evaluateModel(model, testInstances);
   
           return eval.errorRate();
    }
	
	//Version without options
	public static double calculateErrorForModel(String classifierName, String trainPath, String testPath) throws Exception
    {
           return calculateErrorForModel(classifierName, trainPath, testPath, null);
    }
	
	/*
	 * Runs the train/test for a given classifier with a range of options
	 * and returns the best set of parameters
	 * */
	public String[] getBestOptionsForClassifier(String classifierName, TestPair pair, String[] param, int[] initVal, int[] endVal, int[] step) throws Exception{
		assert (param.length == initVal.length);
		assert (param.length == endVal.length);
		assert (param.length == step.length);
		
		String trainPath = pair.getTrainFilePath();
		String testPath = pair.getTestFilePath();
		
		int totalParams = param.length;
		String[] bestOptions = new String[2*totalParams];
		
		for(int i=0; i<totalParams; i++){
			double currentError = Double.MAX_VALUE;
			double lowestError = Double.MAX_VALUE;
			int currentOptValue = initVal[i];
			String[] currentOptions = new String[2*(i+1)];
			for(int k=0; k<(i+1)*2; k++){
				currentOptions[k] = bestOptions[k];
			}
			currentOptions[2*i] = param[i];
			
			while(currentOptValue <= endVal[i]){
				currentOptions[2*i + 1] = String.valueOf(currentOptValue);
				currentError = calculateErrorForModel(classifierName, trainPath, testPath, currentOptions.clone());
				
				//System.out.print("currentOptions: ");
				for(String s : currentOptions){System.out.print(s+" ");};
				System.out.println(": " + currentError);
				
				if(lowestError > currentError){
					lowestError = currentError;
					bestOptions[2*i] = param[i];
					bestOptions[2*i+1] = String.valueOf(currentOptValue);
				}
				currentOptValue += step[i];
			}
		}
		
		System.out.println("Best parameters for "+classifierName+":");
		for(int i=0; i<bestOptions.length; i+=2){
			System.out.println(bestOptions[i]+": "+bestOptions[i+1]);
		}
		
		return bestOptions;
	}
		
		/*
		 * Runs the train/test for a given classifier with a range of options
		 * and returns the best set of parameters
		 * */
	public static String[] getBestOptionsForClassifierForAllDatasets(String classifierName, TestPair[] testPair, String[] param, int[] initVal, int[] endVal, int[] step) throws Exception{
		assert (param.length == initVal.length);
		assert (param.length == endVal.length);
		assert (param.length == step.length);
		
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
					trainPath = testPair[c].getTrainFilePath();
					testPath = testPair[c].getTestFilePath();
					currentAvgError += calculateErrorForModel(classifierName, trainPath, testPath, currentOptions.clone());
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
		
		System.out.println("Best parameters for "+classifierName+":");
		for(int i=0; i<bestOptions.length; i+=2){
			System.out.println(bestOptions[i]+": "+bestOptions[i+1]);
		}
		
		return bestOptions;
	}
	
	
	/**
	 * Serialize the model
	 * @param model
	 * @param outputPath
	 * @throws Exception
	 */
	public static void outputModel(Classifier model, String outputPath) throws Exception
	{
		SerializationHelper.write(outputPath, model);
	}
	
	public static void main(String[] args) throws Exception {
		
		
		String[] classifiers = new String[]{"weka.classifiers.trees.RandomForest", 
											"weka.classifiers.lazy.LWL",
											"weka.classifiers.meta.LogitBoost"};
		
		TestPair[] dataSets = new TestPair[]{new TestPair("anneal","datasets/anneal_train.arff", "datasets/anneal_test.arff"),
											 new TestPair("audiology","datasets/audiology_train.arff", "datasets/audiology_test.arff"),
											 new TestPair("autos","datasets/autos_train.arff", "datasets/autos_test.arff"),
											 new TestPair("balance-scale","datasets/balance-scale_train.arff", "datasets/balance-scale_test.arff"),
											 new TestPair("breast-cancer","datasets/breast-cancer_train.arff", "datasets/breast-cancer_test.arff"),
											 new TestPair("colic","datasets/colic_train.arff", "datasets/colic_test.arff"),
											 new TestPair("credit-a","datasets/credit-a_train.arff", "datasets/credit-a_test.arff"),
											 new TestPair("diabetes","datasets/diabetes_train.arff", "datasets/diabetes_test.arff"),
											 new TestPair("glass","datasets/glass_train.arff", "datasets/glass_test.arff"),
											 new TestPair("heart-c","datasets/heart-c_train.arff", "datasets/heart-c_test.arff"),
											 new TestPair("hepatitis","datasets/hepatitis_train.arff", "datasets/hepatitis_test.arff"),
											 new TestPair("hypothyroid","datasets/hypothyroid_train.arff", "datasets/hypothyroid_test.arff")};
		
		
		/* ------------ MILESTONE 2 ------------- */
		/*
		double currentError, sumOfErrors=0.D, maxError=0.D;
		
		for(int i=0; i<dataSets.length; i++){
			System.out.println("\nDataSet "+(i+1)+"/"+dataSets.length+": "+dataSets[i].getDescription());
			WekaModelBuilder wmb=new WekaModelBuilder();
			currentError = wmb.getBestModelError(classifiers, dataSets[i]);
			sumOfErrors += currentError;
			maxError = maxError<currentError ? currentError : maxError; 
		}
		
		System.out.println("\n average error ratio: " + (sumOfErrors/(double)dataSets.length));
		System.out.println("\n max error ratio: " + maxError);
		*/
		
		
		/* ------------ MILESTONE 3 ------------- */
		
		PriorityQueue<TunedClassifierInfo> tunedClassifiers = new PriorityQueue<TunedClassifierInfo>();
		
		//Classifier 1: Random Forest
		RFTuner rf = new RFTuner();
		TunedClassifierInfo rfModel = rf.getTunedModel(dataSets);
		tunedClassifiers.add(rfModel);
		
		//Classifier 2: Random Forest
		LWLTuner lwl = new LWLTuner();
		TunedClassifierInfo lwlModel = lwl.getTunedModel(dataSets);
		tunedClassifiers.add(lwlModel);
		
		//Classifier 3: Logic Boost
		LogitTuner logit = new LogitTuner();
		TunedClassifierInfo logitModel = logit.getTunedModel(dataSets);
		tunedClassifiers.add(logitModel);
		
		//get best classifier
		TunedClassifierInfo bestClassifier = tunedClassifiers.poll();
		System.out.println("best classifier is: " + bestClassifier.getClassifierName());
		
		//compute model for each dataset using best classifier
		double maxError = Double.MIN_VALUE;
		double sumOfErrors = 0d;
		int i=0;
		for(TestPair tp : dataSets){
			double baselineError=calculateErrorForModel(BASELINE_CLASSIFIER, tp.getTrainFilePath(), tp.getTestFilePath());
			double error = calculateErrorForModel(bestClassifier.getClassifierName(), tp, bestClassifier.getTunedOptions(), true);
			double errorRatio = error / baselineError;
			sumOfErrors += errorRatio;
			System.out.println((i++) + ": " + errorRatio);
			maxError = (errorRatio>maxError) ? errorRatio : maxError;
		}
		System.out.println("average error: " + sumOfErrors/dataSets.length);
		System.out.println("max error: " + maxError);
	}

}
