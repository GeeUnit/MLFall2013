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
		

		TestPair[] dataSetsb = new TestPair[] {
				new TestPair("arrhythmia", "dataset5/arrhythmia_train.arff",
						"dataset5/arrhythmia_test.arff"),
				new TestPair("breast-w", "dataset5/breast-w_train.arff",
						"dataset5/breast-w_test.arff"),
				new TestPair("car", "dataset5/car_train.arff",
						"dataset5/car_test.arff"),
				new TestPair("cmc",	"dataset5/cmc_train.arff",
						"dataset5/cmc_test.arff"),
				new TestPair("credit-g", "dataset5/credit-g_train.arff",
						"dataset5/credit-g_test.arff"),
				new TestPair("cylinder-bands", "dataset5/cylinder-bands_train.arff",
						"dataset5/cylinder-bands_test.arff"),
				new TestPair("dermatology", "dataset5/dermatology_train.arff",
						"dataset5/dermatology_test.arff"),
				new TestPair("ecoli", "dataset5/ecoli_train.arff",
						"dataset5/ecoli_test.arff"),
				new TestPair("flags", "dataset5/flags_train.arff",
						"dataset5/flags_test.arff"),
				new TestPair("haberman", "dataset5/haberman_train.arff",
						"dataset5/haberman_test.arff"),
				new TestPair("heart-h", "dataset5/heart-h_train.arff",
						"dataset5/heart-h_test.arff"),
				new TestPair("heart-statlog", "dataset5/heart-statlog_train.arff",
						"dataset5/heart-statlog_test.arff"),
				new TestPair("kr-vs-kp", "dataset5/kr-vs-kp_train.arff",
						"dataset5/kr-vs-kp_test.arff"),
				new TestPair("liver-disorders", "dataset5/liver-disorders_train.arff",
						"dataset5/liver-disorders_test.arff"),
				new TestPair("mfeat-factors", "dataset5/mfeat-factors_train.arff",
						"dataset5/mfeat-factors_test.arff"),
				new TestPair("mfeat-fourier", "dataset5/mfeat-fourier_train.arff",
						"dataset5/mfeat-fourier_test.arff"),
				new TestPair("mfeat-karhunen", "dataset5/mfeat-karhunen_train.arff",
						"dataset5/mfeat-karhunen_test.arff"),
				new TestPair("primary-tumor", "dataset5/primary-tumor_train.arff",
						"dataset5/primary-tumor_test.arff"),
				new TestPair("sick", "dataset5/sick_train.arff",
						"dataset5/sick_test.arff"),
				new TestPair("sonar", "dataset5/sonar_train.arff",
						"dataset5/sonar_test.arff") };				

		


		double sum = 0D;
		double max = 0D;

		System.out.println(":::::::::::::: RANDOM FOREST ::::::::::::::");
		
		for (TestPair set : dataSetsb) {

			double errorNB = WekaModelBuilder.calculateErrorForModel(
					"weka.classifiers.bayes.NaiveBayes",
					set.getTrainFilePath(), set.getTestFilePath());

			String[] tunedOptions = { "-I", "115", "-K", "4", "-depth", "21" };
			
			URL url = WekaModelBuilder.class.getClassLoader().getResource(
					set.getTrainFilePath());
			DataSource trainSource = new DataSource(url.getFile());

			Instances trainInstances = trainSource.getDataSet();
			trainInstances.setClassIndex(trainInstances.numAttributes() - 1);

			Classifier model;

			// update K according to number of features in data
			double percentageOfFeatures = Double.parseDouble(tunedOptions[3]);
			percentageOfFeatures /= 10d;
			int totalFeatures = trainInstances.numAttributes();
			int usedFeatures = (int) (percentageOfFeatures * (double) totalFeatures);
			tunedOptions[3] = String.valueOf(usedFeatures);
			model = Classifier.forName("weka.classifiers.trees.RandomForest",
					tunedOptions);

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

			FileWriter writer = new FileWriter(new File("5/"
					+ set.getDescription() + "0.predict"));

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

		System.out.println(":::::::::::::: ENSEMBLE ::::::::::::::");
			
		sum = 0D;
		max = 0D;

		for (TestPair set : dataSetsb) {

			double errorNB = WekaModelBuilder.calculateErrorForModel(
					"weka.classifiers.bayes.NaiveBayes",
					set.getTrainFilePath(), set.getTestFilePath());

			EnsembleClassifier ensemble = new EnsembleClassifier();

			URL url = RunMilestone4b.class.getClassLoader().getResource(
					set.getTrainFilePath());
			DataSource trainSource = new DataSource(url.getFile());

			FileWriter writer = new FileWriter(new File("5/"
					+ set.getDescription() + "1.predict"));

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

		System.out.println(sum / dataSetsb.length);
		System.out.println(max);
	}


}
