package edu.cmu.lti.ml.antm.model;

import edu.cmu.lti.ml.antm.data.TestPair;
import weka.classifiers.Classifier;

public interface ModelTuner {
	public Classifier getTunedModel(TestPair[] testPair) throws Exception;
}
