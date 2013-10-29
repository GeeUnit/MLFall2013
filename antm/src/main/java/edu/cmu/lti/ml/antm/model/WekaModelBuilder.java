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
		double bestErrorRatio=1D;
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
			if(testInstances.attribute(38).value((int)model.classifyInstance(current)).equals(current.stringValue(38)))
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
	public void outptuModel(Classifier model, String outputPath) throws Exception
	{
		SerializationHelper.write(outputPath, model);
	}
	
	public static void main(String[] args) throws Exception {
		WekaModelBuilder wmb=new WekaModelBuilder();
		wmb.getBestModel(new String[]{"weka.classifiers.trees.RandomForest", "weka.classifiers.lazy.LWL", "weka.classifiers.meta.LogitBoost" }, new TestPair("datasets/anneal_train.arff", "datasets/anneal_test.arff"));
	}

}
