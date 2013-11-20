package edu.cmu.lti.ml.antm.run;

import java.io.File;
import java.io.FileWriter;
import java.net.URL;

import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;
import edu.cmu.lti.ml.antm.data.TestPair;
import edu.cmu.lti.ml.antm.model.EnsembleClassifier;
import edu.cmu.lti.ml.antm.model.WekaModelBuilder;

public class RunMilestone4b {
	public static void main(String[] args) throws Exception {
		TestPair[] dataSetsb = new TestPair[] {
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
				new TestPair("hypothyroid2",
						"datasets/hypothyroid2_train.arff",
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
				//new TestPair("splice", "datasets/splice_train.arff",
				//		"datasets/splice_test.arff"),
				new TestPair("vehicle", "datasets/vehicle_train.arff",
						"datasets/vehicle_test.arff"),
				new TestPair("vote", "datasets/vote_train.arff",
						"datasets/vote_test.arff"),
				new TestPair("vowel", "datasets/vowel_train.arff",
						"datasets/vowel_test.arff"),
				new TestPair("zoo", "datasets/zoo_train.arff",
						"datasets/zoo_test.arff") };

		for (TestPair set : dataSetsb) {
			EnsembleClassifier ensemble = new EnsembleClassifier();

			URL url = WekaModelBuilder.class.getClassLoader().getResource(
					set.getTrainFilePath());
			DataSource trainSource = new DataSource(url.getFile());

			FileWriter writer = new FileWriter(new File("4b/"
					+ set.getDescription() + ".predict"));

			Instances trainInstances = trainSource.getDataSet();
			trainInstances.setClassIndex(trainInstances.numAttributes() - 1);

			ensemble.buildClassifier(trainInstances);

			url = WekaModelBuilder.class.getClassLoader().getResource(
					set.getTestFilePath());

			ArffLoader testLoader = new ArffLoader();
			testLoader.setURL(url.toString());

			Instances testInstances = testLoader.getStructure();
			int classIndex=testInstances.numAttributes() - 1;
			testInstances.setClassIndex(classIndex);

			Instance current;

			while ((current = testLoader.getNextInstance(testInstances)) != null)
			{
				writer.write(ensemble.classifyInstance(current)+"\n");
			}
			writer.close();
		}
	}
}
