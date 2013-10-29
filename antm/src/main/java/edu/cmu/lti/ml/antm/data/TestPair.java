package edu.cmu.lti.ml.antm.data;

public class TestPair {
	private String trainFilePath;
	private String testFilePath;
	
	public TestPair(String trainFilePath, String testFilePath)
	{
		this.trainFilePath=trainFilePath;
		this.testFilePath=testFilePath;
	}

	public String getTrainFilePath() {
		return trainFilePath;
	}

	public void setTrainFilePath(String trainFilePath) {
		this.trainFilePath = trainFilePath;
	}

	public String getTestFilePath() {
		return testFilePath;
	}

	public void setTestFilePath(String testFilePath) {
		this.testFilePath = testFilePath;
	}
		
}
