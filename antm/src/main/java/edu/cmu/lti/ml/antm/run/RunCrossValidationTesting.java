package edu.cmu.lti.ml.antm.run;

import java.net.URL;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.ReservoirSample;
import edu.cmu.lti.ml.antm.data.TestPair;
import edu.cmu.lti.ml.antm.model.EnsembleClassifier;
import edu.cmu.lti.ml.antm.model.WekaModelBuilder;

public class RunCrossValidationTesting {

	public static Instances shrink(Instances instances) throws Exception {
		ReservoirSample rs = new ReservoirSample();
		rs.setSampleSize(600);
		rs.setInputFormat(instances);

		Instances newTrainInstances = Filter.useFilter(instances, rs);
		newTrainInstances.setClassIndex(newTrainInstances.numAttributes() - 1);
		return newTrainInstances;
	}

	public static void main(String[] args) throws Exception {
		
		//operation selection
		boolean runEnsemble = false;
		boolean runIndividualClassifier = false;
		boolean runDataStatistics = true;
		
		//data selection
		boolean useDevelopmentData = true;
		
		String customClassifierName = "weka.classifiers.trees.RandomForest";
		String[] customClassifierOpts = new String[]{ "-I", "100"};
		int seed = 12345;          	// the seed for randomizing the data
		int folds = 6;         		// the number of folds to generate
		
		TestPair[] dataSets;
		if(useDevelopmentData){
			dataSets = new TestPair[] {
					new TestPair("anneal", "datasets/anneal_train.arff",
							"datasets/anneal_test.arff"),
					new TestPair("audiology", "datasets/audiology_train.arff",
							"datasets/audiology_test.arff"),
					new TestPair("autos", "datasets/autos_train.arff",
							"datasets/autos_test.arff"),
					new TestPair("balance-scale",
							"datasets/balance-scale_train.arff",
							"datasets/balance-scale_test.arff"),
					new TestPair("breast-cancer",
							"datasets/breast-cancer_train.arff",
							"datasets/breast-cancer_test.arff"),
					new TestPair("colic", "datasets/colic_train.arff",
							"datasets/colic_test.arff"),
					new TestPair("credit-a", "datasets/credit-a_train.arff",
							"datasets/credit-a_test.arff"),
					new TestPair("diabetes", "datasets/diabetes_train.arff",
							"datasets/diabetes_test.arff"),
					new TestPair("glass", "datasets/glass_train.arff",
							"datasets/glass_test.arff"),
					new TestPair("heart-c", "datasets/heart-c_train.arff",
							"datasets/heart-c_test.arff"),
					new TestPair("hepatitis", "datasets/hepatitis_train.arff",
							"datasets/hepatitis_test.arff"),
					new TestPair("hypothyroid2", "datasets/hypothyroid2_train.arff",
							"datasets/hypothyroid2_test.arff"),
					new TestPair("ionosphere", "datasets/ionosphere_train.arff",
							"datasets/ionosphere_test.arff"),
					new TestPair("labor", "datasets/labor_train.arff",
							"datasets/labor_test.arff"),
					new TestPair("lymph", "datasets/lymph_train.arff",
							"datasets/lymph_test.arff"),
					new TestPair("mushroom", "datasets/mushroom_train.arff",
							"datasets/mushroom_test.arff"),
					new TestPair("segment", "datasets/segment_train.arff",
							"datasets/segment_test.arff"),
					new TestPair("sonar", "datasets/sonar_train.arff",
							"datasets/sonar_test.arff"),
					new TestPair("soybean", "datasets/soybean_train.arff",
							"datasets/soybean_test.arff"),
					new TestPair("splice", "datasets/splice_train.arff",
							"datasets/splice_test.arff"),
					new TestPair("vehicle", "datasets/vehicle_train.arff",
							"datasets/vehicle_test.arff"),
					new TestPair("vote", "datasets/vote_train.arff",
							"datasets/vote_test.arff"),
					new TestPair("vowel", "datasets/vowel_train.arff",
							"datasets/vowel_test.arff"),
					new TestPair("zoo", "datasets/zoo_train.arff",
							"datasets/zoo_test.arff") };	
		}
		else{
			dataSets = new TestPair[] {
					new TestPair("arrhythmia",
							"dataset5B_Rand/arrhythmia_train.arff",
							"dataset5B_Rand/arrhythmia_test.arff"),
					new TestPair("breast-w", "dataset5B_Rand/breast-w_train.arff",
							"dataset5B_Rand/breast-w_test.arff"),
					new TestPair("car", "dataset5B_Rand/car_train.arff",
							"dataset5B_Rand/car_test.arff"),
					new TestPair("cmc", "dataset5B_Rand/cmc_train.arff",
							"dataset5B_Rand/cmc_test.arff"),
					new TestPair("credit-g", "dataset5B_Rand/credit-g_train.arff",
							"dataset5B_Rand/credit-g_test.arff"),
					new TestPair("cylinder-bands",
							"dataset5B_Rand/cylinder-bands_train.arff",
							"dataset5B_Rand/cylinder-bands_test.arff"),
					new TestPair("dermatology",
							"dataset5B_Rand/dermatology_train.arff",
							"dataset5B_Rand/dermatology_test.arff"),
					new TestPair("ecoli", "dataset5B_Rand/ecoli_train.arff",
							"dataset5B_Rand/ecoli_test.arff"),
					new TestPair("flags", "dataset5B_Rand/flags_train.arff",
							"dataset5B_Rand/flags_test.arff"),
					new TestPair("haberman", "dataset5B_Rand/haberman_train.arff",
							"dataset5B_Rand/haberman_test.arff"),
					new TestPair("heart-h", "dataset5B_Rand/heart-h_train.arff",
							"dataset5B_Rand/heart-h_test.arff"),
					new TestPair("heart-statlog",
							"dataset5B_Rand/heart-statlog_train.arff",
							"dataset5B_Rand/heart-statlog_test.arff"),
					new TestPair("kr-vs-kp", "dataset5B_Rand/kr-vs-kp_train.arff",
							"dataset5B_Rand/kr-vs-kp_test.arff"),
					new TestPair("liver-disorders",
							"dataset5B_Rand/liver-disorders_train.arff",
							"dataset5B_Rand/liver-disorders_test.arff"),
					new TestPair("mfeat-factors",
							"dataset5B_Rand/mfeat-factors_train.arff",
							"dataset5B_Rand/mfeat-factors_test.arff"),
					new TestPair("mfeat-fourier",
							"dataset5B_Rand/mfeat-fourier_train.arff",
							"dataset5B_Rand/mfeat-fourier_test.arff"),
					new TestPair("mfeat-karhunen",
							"dataset5B_Rand/mfeat-karhunen_train.arff",
							"dataset5B_Rand/mfeat-karhunen_test.arff"),
					new TestPair("primary-tumor",
							"dataset5B_Rand/primary-tumor_train.arff",
							"dataset5B_Rand/primary-tumor_test.arff"),
					new TestPair("sick", "dataset5B_Rand/sick_train.arff",
							"dataset5B_Rand/sick_test.arff"),
//					new TestPair("sonar", "dataset5B_Rand/sonar_train.arff",
//							"dataset5B_Rand/sonar_test.arff")};
					new TestPair("spambase", "dataset5B_Rand/spambase_train.arff",
							"dataset5B_Rand/spambase_test.arff") };
		}

		if (runIndividualClassifier){
			System.out.println(":::::::::::::: TUNED CUSTOM CLASSIFIER ::::::::::::::");
			System.out.println(customClassifierName);
			System.out.println("Folds: "+folds);
			
			double sum = 0D;
			double max = 0D;
			
			//for each data set
			for (TestPair dataset : dataSets) {
				
				//do cross-fold cross validation
				double sumOfBaselineError = 0d;
				double sumOfClassifierError = 0d;
				int FVsize=0;
				int trainingSize=0;
				int testingSize=0;
				
				for (int n = 0; n < folds; n++) {
					
					//create instances from train data
					URL url = WekaModelBuilder.class.getClassLoader().getResource( dataset.getTrainFilePath() );
					DataSource trainSource = new DataSource(url.getFile());
					Instances trainInstances = trainSource.getDataSet();
					trainInstances.setClassIndex(trainInstances.numAttributes() - 1);
					
					//create cross-fold validation training and test data
					Instances data = trainInstances;
					Random rand = new Random(seed);   			// create seeded number generator
					Instances randData = new Instances(data);   // create copy of original data
					randData.randomize(rand);         			// randomize data with number generator
					
					Instances train = randData.trainCV(folds, n);
					Instances test = randData.testCV(folds, n);
					
					//baseline error
					double currentBaselineError = WekaModelBuilder.calculateErrorForModel(
							"weka.classifiers.bayes.NaiveBayes",
							train, test, null);
					sumOfBaselineError += currentBaselineError;
					
					//classifier error
					double currentFoldError = WekaModelBuilder.calculateErrorForModel(
							customClassifierName,
							train, test, 
							customClassifierOpts);
					sumOfClassifierError += currentFoldError;
					
					//System.out.println("     "+currentFoldError+"/"+currentBaselineError+"="+(currentFoldError/currentBaselineError));
					
				}//end of cross-fold
				
				double avgErrorRate = sumOfClassifierError / sumOfBaselineError; 
				//System.out.println(avgErrorRate);
				System.out.println(dataset.getDescription()+"\t"+FVsize);
				
				
				sum += avgErrorRate;
				max = avgErrorRate > max ? avgErrorRate : max;
			}//end of data set loop
			
			double totalAvgError = sum / dataSets.length;
			System.out.println(totalAvgError);
			System.out.println(max);
		}
		
		if(runEnsemble){
			System.out.println(":::::::::::::: ENSEMBLE ::::::::::::::");
			System.out.println("Folds: "+folds);
			
			double sum = 0D;
			double max = 0D;
			
			//for each data set
			for (TestPair dataset : dataSets) {
				
				//do cross-fold cross validation
				double sumOfBaselineError = 0d;
				double sumOfClassifierError = 0d;
				
				for (int n = 0; n < folds; n++) {
					
					//create instances from train data
					URL url = WekaModelBuilder.class.getClassLoader().getResource( dataset.getTrainFilePath() );
					DataSource trainSource = new DataSource(url.getFile());
					Instances trainInstances = trainSource.getDataSet();
					trainInstances.setClassIndex(trainInstances.numAttributes() - 1);
					
					//create cross-fold validation training and test data
					Instances data = trainInstances;
					Random rand = new Random(seed);   			// create seeded number generator
					Instances randData = new Instances(data);   // create copy of original data
					randData.randomize(rand);         			// randomize data with number generator
					
					Instances train = randData.trainCV(folds, n);
					Instances test = randData.testCV(folds, n);
					
					//baseline error
					double currentBaselineError = WekaModelBuilder.calculateErrorForModel(
							"weka.classifiers.bayes.NaiveBayes",
							train, test, null);
					sumOfBaselineError += currentBaselineError;
					
					//classifier error
					EnsembleClassifier ensemble = new EnsembleClassifier();
					double currentFoldError = ensemble.calculateError(train, test);
					sumOfClassifierError += currentFoldError;
					
					//System.out.println("     "+currentFoldError+"/"+currentBaselineError+"="+(currentFoldError/currentBaselineError));
					
				}//end of cross-fold
				
				double avgErrorRate = sumOfClassifierError / sumOfBaselineError; 
				System.out.println(avgErrorRate);
				
				sum += avgErrorRate;
				max = avgErrorRate > max ? avgErrorRate : max;
			}//end of data set loop
			
			double totalAvgError = sum / dataSets.length;
			System.out.println(totalAvgError);
			System.out.println(max);
		}
		
		if(runDataStatistics){
			System.out.println(":::::::::::::: DATA STATS ::::::::::::::");
			
			//for each data set
			for (TestPair dataset : dataSets) {
				
				//create instances from train data
				URL url = WekaModelBuilder.class.getClassLoader().getResource( dataset.getTrainFilePath() );
				DataSource trainSource = new DataSource(url.getFile());
				Instances trainInstances = trainSource.getDataSet();
				trainInstances.setClassIndex(trainInstances.numAttributes() - 1);
				
				//create instances from test data
				url = WekaModelBuilder.class.getClassLoader().getResource( dataset.getTestFilePath() );
				DataSource testSource = new DataSource(url.getFile());
				Instances testInstances = testSource.getDataSet();
				testInstances.setClassIndex(testInstances.numAttributes() - 1);
				
				int FVsize = trainInstances.numAttributes() - 1;
				int trainingSize = trainInstances.numInstances();
				int testingSize = testInstances.numInstances();
				
				System.out.println(dataset.getDescription()+"\t"+trainingSize+"\t"+testingSize+"\t"+FVsize);
				
				
			}//end of data set loop
			
		}// end of data stats
		
		
		
	}//end of main method
	
}//end of class
