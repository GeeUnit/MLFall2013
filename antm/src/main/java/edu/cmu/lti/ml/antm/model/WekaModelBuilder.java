package edu.cmu.lti.ml.antm.model;


import java.io.IOException;
import java.net.URL;
import edu.cmu.lti.ml.antm.data.TestPair;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;


public class WekaModelBuilder {
	
	private static String BASELINE_CLASSIFIER="weka.classifiers.bayes.NaiveBayes";

	
	public void getBestModel(String[] modelClassNames, TestPair dataSet) throws Exception
	{
		Classifier baselineModel=trainModel(BASELINE_CLASSIFIER, dataSet.getTrainFilePath());
		double baselineAccuracy=testModel(baselineModel, dataSet.getTestFilePath());
		
		//minimize error ratio
		double bestErrorRatio=Double.MAX_VALUE;
		Classifier bestModel=baselineModel;
		
		for(String classifierName:modelClassNames)
		{
			Classifier model=trainModel(classifierName, dataSet.getTrainFilePath());
			double accuracy=testModel(model, dataSet.getTestFilePath());
			double errorRatio=(1-accuracy)/(1-baselineAccuracy);
			
			if(errorRatio<bestErrorRatio)
			{
				bestErrorRatio=errorRatio;
				bestModel=model;
			}	
			System.out.println(classifierName+"\terror ratio: "+errorRatio);
		}
		
		System.out.println("Best model was: "+bestModel.getClass().getName()+" with error: "+bestErrorRatio);
		this.outputModel(bestModel, dataSet.getDescription());
	}
	
	/**
	 * Train a classifier on a specific training set
	 * @param modelClassName
	 * @param trainSet
	 * @return
	 * @throws Exception
	 */
	public Classifier trainModel(String modelClassName, String trainSet) throws Exception
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
	public double testModel(Classifier model, String testSet) throws IOException, Exception
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
	
	/**
	 * Serialize the model
	 * @param model
	 * @param outputPath
	 * @throws Exception
	 */
	public void outputModel(Classifier model, String outputPath) throws Exception
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
		
		for(int i=0; i<dataSets.length; i++){
			System.out.println("\nDataSet "+(i+1)+"/"+dataSets.length+": "+dataSets[i].getDescription());
			WekaModelBuilder wmb=new WekaModelBuilder();
			wmb.getBestModel(classifiers, dataSets[i]);
		}
		
		
		//WekaModelBuilder wmb=new WekaModelBuilder();
		//wmb.getBestModel(new String[]{"weka.classifiers.trees.RandomForest", "weka.classifiers.lazy.LWL", "weka.classifiers.meta.LogitBoost" }, new TestPair("anneal","datasets/anneal_train.arff", "datasets/anneal_test.arff"));
	}

}
