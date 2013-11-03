package edu.cmu.lti.ml.antm.model;

import edu.cmu.lti.ml.antm.data.TestPair;

public interface ModelTuner {
	public TunedClassifierInfo getTunedModel(TestPair[] testPair) throws Exception;
}
