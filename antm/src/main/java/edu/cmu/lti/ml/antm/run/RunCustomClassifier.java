package edu.cmu.lti.ml.antm.run;

import java.io.File;
import java.io.FileWriter;
import java.net.URL;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.ReservoirSample;
import edu.cmu.lti.ml.antm.data.TestPair;
import edu.cmu.lti.ml.antm.model.WekaModelBuilder;

public class RunCustomClassifier {

	public static Instances shrink(Instances instances) throws Exception {
		ReservoirSample rs = new ReservoirSample();
		rs.setSampleSize(600);
		rs.setInputFormat(instances);

		Instances newTrainInstances = Filter.useFilter(instances, rs);
		newTrainInstances.setClassIndex(newTrainInstances.numAttributes() - 1);
		return newTrainInstances;
	}

	public static void main(String[] args) throws Exception {
		
		String customClassifierName = "weka.classifiers.trees.RandomForest";
		String[] customClassifierOpts = new String[]{ "-I", "115", "-K", "4", "-depth", "21" };
		
		TestPair[] dataSetsa = new TestPair[] {
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
						"datasets/hypothyroid2_test.arff") };				

		
		double sum = 0D;
		double max = 0D;

		System.out.println(":::::::::::::: TUNED CUSTOM CLASSIFIER ::::::::::::::");
		
		for (TestPair set : dataSetsa) {

			double errorNB = WekaModelBuilder.calculateErrorForModel(
					"weka.classifiers.bayes.NaiveBayes",
					set.getTrainFilePath(), set.getTestFilePath());
			
			URL url = WekaModelBuilder.class.getClassLoader().getResource(
					set.getTrainFilePath());
			DataSource trainSource = new DataSource(url.getFile());

			Instances trainInstances = trainSource.getDataSet();
			trainInstances.setClassIndex(trainInstances.numAttributes() - 1);
			
			//update opts K value for random forest according to number of features in data
			if(customClassifierName == "weka.classifiers.trees.RandomForest" && customClassifierOpts.length == 6){
	        	double percentageOfFeatures = Double.parseDouble(customClassifierOpts[3]);
	        	percentageOfFeatures /= 10d;
	        	int totalFeatures = trainInstances.numAttributes();
	        	int usedFeatures = (int) (percentageOfFeatures*(double)totalFeatures);
	        	customClassifierOpts[3] = String.valueOf(usedFeatures);
			}
			

			Classifier model;
			model = Classifier.forName(customClassifierName, customClassifierOpts);

			model.buildClassifier(trainInstances);

			url = RunMilestone4a.class.getClassLoader().getResource(
					set.getTestFilePath());

			ArffLoader testLoader = new ArffLoader();
			testLoader.setURL(url.toString());

			Instances testInstances = testLoader.getStructure();
			int classIndex = testInstances.numAttributes() - 1;
			testInstances.setClassIndex(classIndex);

			Instance current;

			int correct = 0;
			int count = 0;

			FileWriter writer = new FileWriter(new File("5a/"
					+ set.getDescription() + "-LB.predict"));

			while ((current = testLoader.getNextInstance(testInstances)) != null) {
				double label = model.classifyInstance(current);

				if (testInstances.attribute(classIndex).value((int) label)
						.equals(current.stringValue(classIndex))) {
					correct++;
				}
				count++;

				writer.write(label + "\n");

			}
			writer.close();

			double accuracy = correct / (double) count;
			double robustness = (1 - accuracy) / errorNB;
			System.out.println(robustness);
			sum += robustness;
			if (robustness > max) {
				max = robustness;
			}
		}

		System.out.println(sum / dataSetsa.length);
		System.out.println(max);
		
		
		
	}

}
