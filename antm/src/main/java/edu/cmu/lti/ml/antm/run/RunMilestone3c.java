package edu.cmu.lti.ml.antm.run;

import java.net.URL;
import java.text.DecimalFormat;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import edu.cmu.lti.ml.antm.data.TestPair;
import edu.cmu.lti.ml.antm.model.WekaModelBuilder;

public class RunMilestone3c {
	
	public static double calculateErrorForRF(TestPair pair, String[] Options, boolean tuneModel) throws Exception
    {
		
        String testPath = pair.getTestFilePath();
        String trainPath = pair.getTrainFilePath();
		
        URL url=WekaModelBuilder.class.getClassLoader().getResource(trainPath);
        DataSource trainSource = new DataSource(url.getFile());

        Instances trainInstances=trainSource.getDataSet();
        trainInstances.setClassIndex(trainInstances.numAttributes()-1);
        
        Classifier model;
        if(tuneModel){
	        //update K according to number of features in data
        	double percentageOfFeatures = Double.parseDouble(Options[3]);
        	percentageOfFeatures /= 10d;
        	int totalFeatures = trainInstances.numAttributes();
        	int usedFeatures = (int) (percentageOfFeatures*(double)totalFeatures);
	        Options[3] = String.valueOf(usedFeatures);
	        model=Classifier.forName("weka.classifiers.trees.RandomForest", Options);
        }
        else{
        	model=Classifier.forName("weka.classifiers.trees.RandomForest", null);
        }
        model.buildClassifier(trainInstances);

        url=WekaModelBuilder.class.getClassLoader().getResource(testPath);
        DataSource testSource = new DataSource(url.getFile());

        Instances testInstances=testSource.getDataSet();
        testInstances.setClassIndex(testInstances.numAttributes()-1);
       
       
        Evaluation eval=new Evaluation(trainInstances);
        eval.evaluateModel(model, testInstances);
        
        
    	if(tuneModel){
    		outputModel(model, pair.getDescription()+"1.model");
    	}
    	else{
    		outputModel(model, pair.getDescription()+"0.model");
    	}
        

        return eval.errorRate();
    }
	
	public static double calculateBaselineError(TestPair pair) throws Exception
    {
		
        String testPath = pair.getTestFilePath();
        String trainPath = pair.getTrainFilePath();
		
        URL url=WekaModelBuilder.class.getClassLoader().getResource(trainPath);
        DataSource trainSource = new DataSource(url.getFile());

        Instances trainInstances=trainSource.getDataSet();
        trainInstances.setClassIndex(trainInstances.numAttributes()-1);
        
        Classifier model = Classifier.forName("weka.classifiers.bayes.NaiveBayes", null);
        model.buildClassifier(trainInstances);

        url = WekaModelBuilder.class.getClassLoader().getResource(testPath);
        DataSource testSource = new DataSource(url.getFile());

        Instances testInstances = testSource.getDataSet();
        testInstances.setClassIndex(testInstances.numAttributes()-1);
       
       
        Evaluation eval = new Evaluation(trainInstances);
        eval.evaluateModel(model, testInstances);

        return eval.errorRate();
    }
	
	public static void outputModel(Classifier model, String outputPath) throws Exception
	{
		SerializationHelper.write(outputPath, model);
	}

	
	public static void main(String[] args) throws Exception {
		
		TestPair[] dataSets = 
				new TestPair[]{new TestPair("anneal","datasets/anneal_train.arff", "datasets/anneal_test.arff"),
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
				new TestPair("hypothyroid","datasets/hypothyroid2_train.arff", "datasets/hypothyroid2_test.arff")};

		//K parameter will be added later because is dependent on dataset -> 40% of total features 
		String[] tunedOptions = {"-I", "115", "-K", "4", "-depth", "21"}; 
		
		//Compute results
		double maxErrorUntuned = Double.MIN_VALUE;
		double maxErrorTuned = Double.MIN_VALUE;
		double sumOfErrorsUntuned = 0d;
		double sumOfErrorsTuned = 0d;
		int i=1;
		
		//Untuned Model
		System.out.println("UNTUNED MODELS");
		for(TestPair tp : dataSets){
			double baselineError = calculateBaselineError(tp);
			double errorUntuned = calculateErrorForRF(tp, tunedOptions.clone(), false);
			double errorRatioUntuned = errorUntuned / baselineError;
			sumOfErrorsUntuned += errorRatioUntuned;
			System.out.println((i++) + ": " + errorRatioUntuned);
			maxErrorUntuned = (errorRatioUntuned>maxErrorUntuned) ? errorRatioUntuned : maxErrorUntuned;
		}
		System.out.println("average error: " + sumOfErrorsUntuned/dataSets.length);
		System.out.println("max error: " + maxErrorUntuned);
		
		//Tuned Model
		System.out.println("\nTUNED MODELS");
		i=1;
		for(TestPair tp : dataSets){
			double baselineError = calculateBaselineError(tp);
			double errorTuned = calculateErrorForRF(tp, tunedOptions.clone(), true);
			double errorRatioTuned = errorTuned / baselineError;
			sumOfErrorsTuned += errorRatioTuned;
			System.out.println((i++) + ": " + errorRatioTuned);
			maxErrorTuned = (errorRatioTuned>maxErrorTuned) ? errorRatioTuned : maxErrorTuned;
		}
		System.out.println("average error: " + sumOfErrorsTuned/dataSets.length);
		System.out.println("max error: " + maxErrorTuned);
		
		DecimalFormat df = new DecimalFormat("#.##");
		System.out.println("\n\nimprovement on avg error: "+df.format((sumOfErrorsUntuned-sumOfErrorsTuned)*100d/sumOfErrorsUntuned) + "%");
		System.out.println("improvement on max error: "+df.format((maxErrorUntuned-maxErrorTuned)*100d/maxErrorUntuned) + "%");
	}

}
