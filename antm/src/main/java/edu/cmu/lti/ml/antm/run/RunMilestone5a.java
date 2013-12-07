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
import edu.cmu.lti.ml.antm.model.EnsembleClassifier;
import edu.cmu.lti.ml.antm.model.WekaModelBuilder;

public class RunMilestone5a {


	public static Instances shrink(Instances instances) throws Exception {
		ReservoirSample rs = new ReservoirSample();
		rs.setSampleSize(600);
		rs.setInputFormat(instances);

		Instances newTrainInstances = Filter.useFilter(instances, rs);
		newTrainInstances.setClassIndex(newTrainInstances.numAttributes() - 1);
		return newTrainInstances;
	}

	public static void main(String[] args) throws Exception {
		

		TestPair[] dataSetsa = new TestPair[] {
				new TestPair("anneal", "dataset5A_rand/anneal_train.arff",
						"dataset5A_rand/anneal_test.arff"),
				new TestPair("audiology", "dataset5A_rand/audiology_train.arff",
						"dataset5A_rand/audiology_test.arff"),
				new TestPair("autos", "dataset5A_rand/autos_train.arff",
						"dataset5A_rand/autos_test.arff"),
				new TestPair("balance-scale",
						"dataset5A_rand/balance-scale_train.arff",
						"dataset5A_rand/balance-scale_test.arff"),
				new TestPair("breast-cancer",
						"dataset5A_rand/breast-cancer_train.arff",
						"dataset5A_rand/breast-cancer_test.arff"),
				new TestPair("colic", "dataset5A_rand/colic_train.arff",
						"dataset5A_rand/colic_test.arff"),
				new TestPair("credit-a", "dataset5A_rand/credit-a_train.arff",
						"dataset5A_rand/credit-a_test.arff"),
				new TestPair("diabetes", "dataset5A_rand/diabetes_train.arff",
						"dataset5A_rand/diabetes_test.arff"),
				new TestPair("glass", "dataset5A_rand/glass_train.arff",
						"dataset5A_rand/glass_test.arff"),
				new TestPair("heart-c", "dataset5A_rand/heart-c_train.arff",
						"dataset5A_rand/heart-c_test.arff"),
				new TestPair("hepatitis", "dataset5A_rand/hepatitis_train.arff",
						"dataset5A_rand/hepatitis_test.arff"),
				new TestPair("hypothyroid2", "dataset5A_rand/hypothyroid2_train.arff",
						"dataset5A_rand/hypothyroid2_test.arff"),
				new TestPair("ionosphere", "dataset5A_rand/ionosphere_train.arff",
						"dataset5A_rand/ionosphere_test.arff"),
				new TestPair("labor", "dataset5A_rand/labor_train.arff",
						"dataset5A_rand/labor_test.arff"),
				new TestPair("lymph", "dataset5A_rand/lymph_train.arff",
						"dataset5A_rand/lymph_test.arff"),
				new TestPair("mushroom", "dataset5A_rand/mushroom_train.arff",
						"dataset5A_rand/mushroom_test.arff"),
				new TestPair("segment", "dataset5A_rand/segment_train.arff",
						"dataset5A_rand/segment_test.arff"),
				new TestPair("sonar", "dataset5A_rand/sonar_train.arff",
						"dataset5A_rand/sonar_test.arff"),
				new TestPair("soybean", "dataset5A_rand/soybean_train.arff",
						"dataset5A_rand/soybean_test.arff"),
				new TestPair("splice", "dataset5A_rand/splice_train.arff",
						"dataset5A_rand/splice_test.arff"),
				new TestPair("vehicle", "dataset5A_rand/vehicle_train.arff",
						"dataset5A_rand/vehicle_test.arff"),
				new TestPair("vote", "dataset5A_rand/vote_train.arff",
						"dataset5A_rand/vote_test.arff"),
				new TestPair("vowel", "dataset5A_rand/vowel_train.arff",
						"dataset5A_rand/vowel_test.arff"),
				new TestPair("zoo", "dataset5A_rand/zoo_train.arff",
						"dataset5A_rand/zoo_test.arff") };				

		


		double sum = 0D;
		double max = 0D;

		System.out.println(":::::::::::::: UNTUNED RANDOM FOREST ::::::::::::::");
		
		for (TestPair set : dataSetsa) {

			double errorNB = WekaModelBuilder.calculateErrorForModel(
					"weka.classifiers.bayes.NaiveBayes",
					set.getTrainFilePath(), set.getTestFilePath());
			
			URL url = WekaModelBuilder.class.getClassLoader().getResource(
					set.getTrainFilePath());
			DataSource trainSource = new DataSource(url.getFile());

			Instances trainInstances = trainSource.getDataSet();
			trainInstances.setClassIndex(trainInstances.numAttributes() - 1);

			Classifier model;
			model = Classifier.forName("weka.classifiers.trees.RandomForest", null);

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
		
		
		System.out.println("::::::::: ENSEMBLE ::::::::::::::");

		sum = 0D;
		max = 0D;

		for (TestPair set : dataSetsa) {

			double errorNB = WekaModelBuilder.calculateErrorForModel(
					"weka.classifiers.bayes.NaiveBayes",
					set.getTrainFilePath(), set.getTestFilePath());

			EnsembleClassifier ensemble = new EnsembleClassifier();

			URL url = RunMilestone4b.class.getClassLoader().getResource(
					set.getTrainFilePath());
			DataSource trainSource = new DataSource(url.getFile());

			FileWriter writer = new FileWriter(new File("5a/"
					+ set.getDescription() + "-L5.predict"));

			Instances trainInstances = trainSource.getDataSet();

			trainInstances.setClassIndex(trainInstances.numAttributes() - 1);

			if (set.getDescription().equals("splice")) {
				trainInstances = shrink(trainInstances);
			}
			ensemble.buildClassifier(trainInstances);

			url = RunMilestone4b.class.getClassLoader().getResource(
					set.getTestFilePath());

			ArffLoader testLoader = new ArffLoader();
			testLoader.setURL(url.toString());

			Instances testInstances = testLoader.getStructure();
			int classIndex = testInstances.numAttributes() - 1;
			testInstances.setClassIndex(classIndex);

			Instance current;

			int correct = 0;
			int count = 0;

			while ((current = testLoader.getNextInstance(testInstances)) != null) {
				double label = ensemble.classifyInstance(current);

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
