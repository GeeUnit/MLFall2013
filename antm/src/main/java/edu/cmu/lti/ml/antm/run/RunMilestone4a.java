package edu.cmu.lti.ml.antm.run;

import java.net.URL;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import edu.cmu.lti.ml.antm.data.TestPair;
import edu.cmu.lti.ml.antm.model.EnsembleClassifier;
import edu.cmu.lti.ml.antm.model.WekaModelBuilder;

public class RunMilestone4a {
	
	public static void main(String[] args) throws Exception
	{
		TestPair[] dataSetsa = new TestPair[]{new TestPair("anneal","datasets/anneal_train.arff", "datasets/anneal_test.arff"),
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
				 new TestPair("hypothyroid2","datasets/hypothyroid2_train.arff", "datasets/hypothyroid2_test.arff")};
		
		double max=0D;
		double sum=0D;
		for(TestPair set:dataSetsa)
		{
			double errorNB=WekaModelBuilder.calculateErrorForModel("weka.classifiers.bayes.NaiveBayes", set.getTrainFilePath(), set.getTestFilePath());
			
			EnsembleClassifier ensemble=new EnsembleClassifier();

	        URL url=WekaModelBuilder.class.getClassLoader().getResource(set.getTrainFilePath());
	        DataSource trainSource = new DataSource(url.getFile());

	        Instances trainInstances=trainSource.getDataSet();
	        trainInstances.setClassIndex(trainInstances.numAttributes()-1);
	        
	        ensemble.buildClassifier(trainInstances);

	        url=WekaModelBuilder.class.getClassLoader().getResource(set.getTestFilePath());
	        DataSource testSource = new DataSource(url.getFile());

	        Instances testInstances=testSource.getDataSet();
	        testInstances.setClassIndex(testInstances.numAttributes()-1);
	       
	       
	        Evaluation eval=new Evaluation(trainInstances);
	       
	        eval.evaluateModel(ensemble, testInstances);
	       
			double errorVote=eval.errorRate();
			double robustness=errorVote/errorNB;
			System.out.println(robustness);
			WekaModelBuilder.outputModel(ensemble, "4a/"+set.getDescription()+".model");
			if(robustness>max)
			{
				max=robustness;
			}
			sum+=robustness;
		}
		
		System.out.println((sum/dataSetsa.length));
		System.out.println(max);
	}
}
