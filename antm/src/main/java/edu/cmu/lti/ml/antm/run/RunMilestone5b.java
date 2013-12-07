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

public class RunMilestone5b {

	public static Instances shrink(Instances instances) throws Exception {
		ReservoirSample rs = new ReservoirSample();
		rs.setSampleSize(600);
		rs.setInputFormat(instances);

		Instances newTrainInstances = Filter.useFilter(instances, rs);
		newTrainInstances.setClassIndex(newTrainInstances.numAttributes() - 1);
		return newTrainInstances;
	}

	public static void main(String[] args) throws Exception {
		TestPair[] dataSetsb = new TestPair[] {
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
//				new TestPair("sonar", "dataset5B_Rand/sonar_train.arff",
//						"dataset5B_Rand/sonar_test.arff"),
				new TestPair("spambase", "dataset5B_Rand/spambase_train.arff",
						"dataset5B_Rand/spambase_test.arff") };

		double sum = 0D;
		double max = 0D;

		for (TestPair set : dataSetsb) {
	
			System.out.println(set.getDescription());
			double errorNB = WekaModelBuilder.calculateErrorForModel(
					"weka.classifiers.bayes.NaiveBayes",
					set.getTrainFilePath(), set.getTestFilePath());

			URL url = WekaModelBuilder.class.getClassLoader().getResource(
					set.getTrainFilePath());
			DataSource trainSource = new DataSource(url.getFile());

			Instances trainInstances = trainSource.getDataSet();
			trainInstances.setClassIndex(trainInstances.numAttributes() - 1);

			Classifier model;

			model = Classifier.forName("weka.classifiers.trees.RandomForest",
					null);

			model.buildClassifier(trainInstances);

			url = RunMilestone5b.class.getClassLoader().getResource(
					set.getTestFilePath());

			ArffLoader testLoader = new ArffLoader();
			testLoader.setURL(url.toString());

			Instances testInstances = testLoader.getStructure();
			int classIndex = testInstances.numAttributes() - 1;
			testInstances.setClassIndex(classIndex);

			Instance current;

			int correct = 0;
			int count = 0;

			FileWriter writer = new FileWriter(new File("5b/"
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

		System.out.println(sum / dataSetsb.length);
		System.out.println(max);

		System.out.println("::::::::: ENSEMBLE ::::::::::::::");

		sum = 0D;
		max = 0D;

		for (TestPair set : dataSetsb) {

			double errorNB = WekaModelBuilder.calculateErrorForModel(
					"weka.classifiers.bayes.NaiveBayes",
					set.getTrainFilePath(), set.getTestFilePath());

			EnsembleClassifier ensemble = new EnsembleClassifier();

			URL url = RunMilestone5b.class.getClassLoader().getResource(
					set.getTrainFilePath());
			DataSource trainSource = new DataSource(url.getFile());

			FileWriter writer = new FileWriter(new File("5b/"
					+ set.getDescription() + "-L5.predict"));

			Instances trainInstances = trainSource.getDataSet();

			trainInstances.setClassIndex(trainInstances.numAttributes() - 1);

			// if (set.getDescription().equals("splice")) {
			// trainInstances = shrink(trainInstances);
			// }
			ensemble.buildClassifier(trainInstances);

			url = RunMilestone5b.class.getClassLoader().getResource(
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

		System.out.println(sum / dataSetsb.length);
		System.out.println(max);
	}
}
